import collections
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from modules.blocks import MLP
from modules.tracker import PubTracker as Tracker
from efg.modeling.operators import nms_gpu, boxes_iou3d_gpu
from modules.utils import Instances
from modules.utils import (
    rotate_points_along_z,
    encode_boxes_res_torch,
    decode_torch,
    transform_trajs_to_global_coords,
    transform_trajs_to_local_coords,
    get_corner_points_of_roi,
    spherical_coordinate,
    reorder_rois,
    crop_current_frame_points,
    transform_box_to_global,
    transform_global_to_current_torch,
)
from losses import WeightedSmoothL1Loss, get_corner_loss
from transformer import (
    TransformerEncoder,
    TransformerEncoderGlobalLocal,
    TransformerEncoderLayer,
    TransformerEncoderLayerGlobalLocal,
)

from pointnet import PointNet, MotionEncoder


class TrajectoryFormer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = torch.device(config.model.device)
        self.config = config
        self.is_train = config.task == "train"
        self.hidden_dim = config.model.hidden_dim

        self.seqboxembed = PointNet(config.model.boxes_dim, channels=self.hidden_dim)
        self.velboxembed = MotionEncoder(
            config.model.motion_input_dim,
            self.hidden_dim,
            out_channels=3 * config.model.motion_pred_frames,
        )

        self.traj_length = config.dataset.traj_length
        self.num_lidar_points = config.model.num_lidar_points
        self.num_hypo_det = config.model.num_hypo_det
        self.num_hypo_pred = config.model.num_hypo_pred
        self.num_hypo_train = (self.num_hypo_pred + self.num_hypo_det) * 2
        self.reg_loss_func = WeightedSmoothL1Loss(code_weights=None)

        self.point_reg = MLP(self.hidden_dim, self.hidden_dim, 7, 3)
        self.joint_cls = MLP(self.hidden_dim, self.hidden_dim, 1, 3)
        self.point_cls = MLP(self.hidden_dim, self.hidden_dim, 1, 3)
        self.boxes_cls = MLP(self.hidden_dim, self.hidden_dim, 1, 3)
        self.cls_embed = MLP(
            self.hidden_dim * 2 + 3, self.hidden_dim, self.hidden_dim, 3
        )
        self.up_dimension_geometry = MLP(
            config.model.point_dim, self.hidden_dim, self.hidden_dim, 3
        )
        self.dist_thresh = config.model.dist_thresh

        self.token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.token_traj = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.num_encoder_layers = config.model.enc_layers
        self.dim_feedforward = config.model.dim_feedforward
        self.nhead = config.model.nhead
        encoder_layer = [
            TransformerEncoderLayer(
                self.config,
                d_model=self.hidden_dim,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
            )
            for i in range(self.num_encoder_layers)
        ]
        encoder_layer_gl = [
            TransformerEncoderLayerGlobalLocal(
                self.config,
                d_model=self.hidden_dim,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
            )
            for i in range(self.num_encoder_layers)
        ]
        encoder_norm = None
        self.encoder_fg = TransformerEncoder(
            encoder_layer, self.num_encoder_layers, encoder_norm, self.config
        )
        self.encoder_globallocal = TransformerEncoderGlobalLocal(
            encoder_layer_gl, self.num_encoder_layers, encoder_norm, self.config
        )
        self.car_embed = torch.tensor([1, 0, 0]).cuda().float().reshape(1, 1, 3)
        self.ped_embed = torch.tensor([0, 1, 0]).cuda().float().reshape(1, 1, 3)
        self.cyc_embed = torch.tensor([0, 0, 1]).cuda().float().reshape(1, 1, 3)
        self.train_nms_thresh = self.config.dataset.nms_thresh
        self.train_score_thresh = self.config.dataset.score_thresh
        # eval
        self.max_id = 0
        self.WAYMO_TRACKING_NAMES = config.dataset.classes
        self.nms_thresh = self.config.model.nms_thresh
        max_dist = config.model.max_dist
        self.num_hypo_inference = config.model.num_hypo_pred_eval
        self.history_traj_frames = config.model.history_frames_eval
        self.keep_thresh_car = config.model.track_score.car
        self.keep_thresh_ped = config.model.track_score.ped
        self.keep_thresh_cyc = config.model.track_score.cyc
        self.new_born_car = config.model.new_born_score.car
        self.new_born_ped = config.model.new_born_score.ped
        self.new_born_cyc = config.model.new_born_score.cyc
        self.new_born_nms_thresh = self.config.model.new_born_nms_thresh
        self.to(self.device)
        self.tracker = Tracker(max_dist=max_dist)
        self.load_motion_module = False

    def forward(self, batched_inputs):
        if not self.load_motion_module:
            self.load_pretrain_motionencoder()

        if self.is_train:
            loss_dict = self.forward_train(batched_inputs)
            return loss_dict
        else:
            results = self.forward_inference(batched_inputs)
            return results

    def forward_train(self, batched_inputs):
        self.batch_size = len(batched_inputs)

        samples = collate([bi[0] for bi in batched_inputs], self.device)

        targets = [bi[1]["annotations"] for bi in batched_inputs]
        for key in ["gt_boxes", "difficulty", "num_points_in_gt", "labels"]:
            for i in range(self.batch_size):
                targets[i][key] = torch.tensor(targets[i][key], device=self.device)

        if self.is_train:
            load_boxes3d = [
                torch.from_numpy(bi[1]["annotations"]["pred_boxes3d"]).cuda()
                for bi in batched_inputs
            ]
            load_scores = [
                torch.from_numpy(bi[1]["annotations"]["pred_scores"]).cuda()
                for bi in batched_inputs
            ]
            load_labels = [
                torch.from_numpy(bi[1]["annotations"]["pred_labels"]).cuda()
                for bi in batched_inputs
            ]

            pred_boxes3d, pred_labels, det_boxes3d, traj = self.organize_proposals(
                load_boxes3d, load_scores, load_labels
            )
            self.num_track = pred_boxes3d.shape[1]
            if self.num_track > 0 and det_boxes3d.shape[1] > 0:
                loss_dict = {}
                hypotheses_aug = self.hypotheses_augment(pred_boxes3d, targets)

                (
                    global_trajectory_hypothses,
                    global_candidates,
                ) = self.generate_trajectory_hypothses(
                    pred_boxes3d, det_boxes3d, traj, self.num_hypo_det, hypotheses_aug
                )

                point_feat_list = self.get_trajcetory_point_feature(
                    global_trajectory_hypothses, samples
                )
                point_cls = self.point_cls(torch.cat(point_feat_list, 0)).squeeze(-1)

                boxes_feat = self.get_trajectory_boxes_feature(
                    global_trajectory_hypothses
                )
                boxes_cls = self.boxes_cls(boxes_feat).reshape(-1, self.num_hypo_train)

                hypotheses_feat = self.get_trajectory_hypotheses_feat(
                    point_feat_list, boxes_feat, pred_labels
                )
                feat_list = self.encoder_globallocal(hypotheses_feat)

                joint_cls_list = []
                for i in range(self.num_encoder_layers):
                    joint_cls = (
                        self.joint_cls(feat_list[i])
                        .squeeze(-1)
                        .reshape(-1, self.num_hypo_train)
                    )
                    joint_cls_list.append(joint_cls)
                joint_cls = torch.cat(joint_cls_list, 0)
                point_reg = self.point_reg(torch.cat(point_feat_list, 0)).reshape(
                    3, self.batch_size * self.num_track, self.num_hypo_train, 7
                )
                point_reg = point_reg.reshape(1, -1, 7)
                fg_iou_mask, fg_reg_mask, ious_targets, gt_boxes = self.get_cls_targets(
                    pred_boxes3d, global_candidates, targets
                )
                rois = global_candidates[..., :7].reshape(-1, 7)
                reg_targets = self.get_reg_targets(rois, gt_boxes)

                (
                    loss_cls_sum,
                    loss_reg_sum,
                ) = self.get_loss(
                    rois,
                    gt_boxes,
                    point_cls,
                    joint_cls,
                    boxes_cls,
                    point_reg,
                    ious_targets,
                    reg_targets,
                    fg_reg_mask,
                    fg_iou_mask,
                )

                if gt_boxes.shape[0] > 0:
                    loss_dict.update(
                        {
                            "loss_cls": loss_cls_sum,
                            "loss_reg": loss_reg_sum,
                        }
                    )

                else:
                    loss_dict = {
                        "loss_cls": loss_cls_sum,
                        "loss_reg": torch.tensor([0.0]).cuda().reshape(1, -1),
                    }

            else:
                loss_dict = {
                    "loss_cls": torch.tensor([0.0]).cuda().reshape(1, -1),
                    "loss_reg": torch.tensor([0.0]).cuda().reshape(1, -1),
                }

            return loss_dict

    def forward_inference(self, batched_inputs):
        self.batch_size = len(batched_inputs)

        samples = collate([bi[0] for bi in batched_inputs], self.device)
        self.frame_id = int(
            batched_inputs[0][1]["token"].split("_frame_")[-1].split(".")[0]
        )

        det_boxes3d = [
            torch.from_numpy(bi[1]["annotations"]["pred_boxes3d"]).cuda()
            for bi in batched_inputs
        ][0][None]
        det_scores = [
            torch.from_numpy(bi[1]["annotations"]["pred_scores"]).cuda()
            for bi in batched_inputs
        ][0][None]
        det_labels = [
            torch.from_numpy(bi[1]["annotations"]["pred_labels"]).cuda()
            for bi in batched_inputs
        ][0][None]

        pose = batched_inputs[0][1]["veh_to_global"]
        self.pose = pose
        det_boxes = det_boxes3d[:, :, [0, 1, 2, 3, 4, 5, 8]]
        det_vels = det_boxes3d[:, :, [6, 7]]

        if self.frame_id == 0:
            results = []
            track_out, instance, global_boxes, global_vels = self.init_trajectory(
                pose, det_boxes, det_scores, det_vels, det_labels
            )
            results.append(track_out)
            tracks = []
            for i in range(instance.pred_boxes.shape[0]):
                tracks.append(
                    {
                        "translation": global_boxes[i, :2].cpu().numpy(),
                        "ct": global_boxes[i, :2].cpu().numpy(),
                        "velocity": global_vels[i].cpu().numpy(),
                        "detection_name": self.WAYMO_TRACKING_NAMES[
                            int(det_labels[0, i] - 1)
                        ],
                        "score": instance.scores[i].cpu().numpy(),
                        "box_id": instance.track_id[i].cpu().numpy(),
                        "tracking_id": instance.track_id[i].cpu().numpy(),
                        "label_preds": instance.pred_classes[i].cpu().numpy(),
                        "active": 1,
                        "age": 1,
                    }
                )
            self.tracker.reset(self.max_id, tracks)
            return results
        else:
            keep = self.class_agnostic_nms(
                det_boxes[0], det_scores[0].reshape(-1), nms_thresh=self.nms_thresh
            )
            det_boxes = det_boxes[:, keep]
            det_vels = det_vels[:, keep]
            det_scores = det_scores[:, keep]
            det_labels = det_labels[:, keep]

            if self.instances[-1].track_id.shape[0] == 0:
                if (det_boxes.sum(-1) == 0).all():
                    track_out = {
                        "track_scores": torch.zeros(0),
                        "track_labels": torch.zeros(0),
                        "track_boxes3d": torch.zeros(0, 7),
                        "track_ids": torch.zeros(0).int(),
                    }
                    results = []
                    results.append(track_out)
                    return results
                else:
                    track_out = self.init_trajectory(
                        pose, det_boxes, det_scores, det_vels, det_labels
                    )[0]
                    results = []
                    results.append(track_out)
                    return results
            cur_ids = self.instances[-1].track_id
            traj, traj_vels = self.get_history_traj(cur_ids)
            self.num_track, self.num_candi = traj.shape[2], traj.shape[3]

            (
                points,
                trajectory_hypothese,
                global_candidates,
                joint_vels,
                asso_mask,
            ) = self.get_point_and_trajectory(
                traj[:, : self.history_traj_frames - 1],
                traj_vels[:, : self.history_traj_frames - 1],
                samples,
                det_boxes,
                det_vels,
                det_labels,
                det_scores,
            )

            point_feat = self.get_proposal_aware_point_feature(
                points.reshape(-1, points.shape[-2], points.shape[-1]),
                trajectory_hypothese[:, 0].reshape(self.batch_size, 1, -1, 8),
                self.num_track * self.num_candi,
            )
            point_feat = point_feat.reshape(
                -1, self.num_lidar_points, point_feat.shape[-1]
            )
            boxes_feat = self.get_trajectory_boxes_feature(
                trajectory_hypothese[:, : self.history_traj_frames]
            )
            token = self.token.repeat(
                self.batch_size * (self.num_track * self.num_candi), 1, 1
            ).cuda()
            point_feat_list = self.encoder_fg(token, point_feat)
            fg_confidence = self.point_cls(point_feat_list[-1]).reshape(
                self.num_track, self.num_candi
            )
            fg_confidence = fg_confidence.sigmoid()
            hypotheses_feat = self.get_trajectory_hypotheses_feat_inference(
                point_feat_list, boxes_feat, self.instances[-1].pred_classes
            )
            feat_list = self.encoder_globallocal(hypotheses_feat)
            hypotheses_scores = (
                self.joint_cls(feat_list[-1]).reshape(-1, self.num_candi).sigmoid()
            )
            point_reg = self.point_reg(point_feat_list[-1]).reshape(
                self.batch_size * self.num_track, -1, 7
            )
            point_pred = self.generate_refined_boxes(
                global_candidates[..., :7], box_preds=point_reg
            )
            refined_candidates = point_pred.reshape(self.num_track, self.num_candi, 7)
            keep_mask = self.get_keep_mask(fg_confidence, asso_mask)

            output_new = {
                "pred_logits": det_scores,
                "pred_boxes": det_boxes,
                "pred_labels": det_labels,
                "pred_vels": det_vels,
            }

            selected = hypotheses_scores.max(-1)[1][keep_mask]
            joint_scores = fg_confidence
            matched_boxes = global_candidates[keep_mask, selected][..., :7]
            refined_matched_boxes = refined_candidates[keep_mask, selected]
            matched_vels = joint_vels[keep_mask, selected]
            matched_scores = joint_scores[keep_mask, selected].reshape(-1)
            matched_labels = self.instances[-1].pred_classes[keep_mask]
            track_id = cur_ids[keep_mask]

            track_new = {
                "matched_boxes": matched_boxes,
                "refined_matched_boxes": refined_matched_boxes,
                "matched_vels": matched_vels,
                "matched_scores": matched_scores,
                "matched_labels": matched_labels,
                "track_id": track_id,
            }

            track_out = self.update_trajectory(output_new, track_new)

            results = []
            results.append(track_out)

            return results

    def get_history_traj(self, cur_ids):
        num_frames = self.num_hypo_inference + self.history_traj_frames
        window_list = self.instances[::-1][:num_frames]
        traj = torch.zeros(1, len(window_list), cur_ids.shape[0], 7).cuda()
        traj_vels = torch.zeros(1, len(window_list), cur_ids.shape[0], 2).cuda()
        pose_cur_cuda = torch.from_numpy(self.pose).cuda().float()
        for k, id in enumerate(cur_ids):
            traj_id = self.history_trajectory_bank[id.item()]
            boxes_cat = (
                torch.cat(
                    [x for t, x in enumerate(traj_id["track_boxes3d"][:num_frames])],
                    dim=0,
                )
                .reshape(-1, 7)
                .clone()
            )
            vels_cat = (
                torch.cat(
                    [x for t, x in enumerate(traj_id["track_vels"][:num_frames])], dim=0
                )
                .reshape(-1, 2)
                .clone()
            )
            transfered_traj, transfered_vel = transform_global_to_current_torch(
                boxes_cat, vels_cat, pose_cur_cuda)
            traj[0, : boxes_cat.shape[0], k] = transfered_traj
            traj_vels[0, : vels_cat.shape[0], k] = transfered_vel

        return traj, traj_vels

    def load_pretrain_motionencoder(self):
        ckpt = torch.load(self.config.dataset.motion_model, map_location="cpu")
        if "model" in ckpt.keys():
            ckpt = ckpt["model"]
        motion_module_name = "velboxembed"
        ckpt_traj = {}
        for k, v in ckpt.items():
            if motion_module_name in k:
                ckpt_traj[k.replace("velboxembed.", "")] = v

        self.velboxembed.load_state_dict(ckpt_traj, True)
        for parm in self.velboxembed.parameters():
            parm.required_grad = False
        self.velboxembed.eval()
        self.load_motion_module = True

    def hypotheses_augment(self, batch_bbox, targets):
        range_config = [
            [0.5, 0.1, np.pi / 12, 0.7],
            [0.5, 0.15, np.pi / 12, 0.7],
            [0.5, 0.15, np.pi / 9, 0.5],
            [0.5, 0.15, np.pi / 6, 0.3],
            [0.5, 0.15, np.pi / 3, 0.2],
        ]

        max_aug_iteration = 20
        aug_list = []
        for bs_idx in range(batch_bbox.shape[0]):
            bbox = batch_bbox[bs_idx]
            aug_list_batch = []
            count = 0
            for _ in range(max_aug_iteration):
                idx = np.random.randint(low=0, high=len(range_config), size=(1,))[0]
                pos_shift = (
                    torch.from_numpy(
                        ((np.random.rand(3) - 0.5) / 0.5) * range_config[idx][0]
                    )
                    .cuda()
                    .float()
                )
                hwl_scale = (
                    torch.from_numpy(
                        ((np.random.rand(3) - 0.5) / 0.5) * range_config[idx][1] + 1.0
                    )
                    .cuda()
                    .float()
                )
                angle_rot = (
                    torch.from_numpy(
                        ((np.random.rand(1) - 0.5) / 0.5) * range_config[idx][2]
                    )
                    .cuda()
                    .float()
                )
                aug_box3d = torch.cat(
                    [
                        bbox[:, 0:3] + pos_shift[None, :],
                        bbox[:, 3:6] * hwl_scale[None, :],
                        bbox[:, 6:7] + angle_rot[None, :],
                    ],
                    -1,
                ).cuda()
                if aug_box3d.shape[0] > 0 and targets[bs_idx]["gt_boxes"].shape[0] > 0:
                    ious = boxes_iou3d_gpu(
                        aug_box3d.float(),
                        targets[bs_idx]["gt_boxes"][:, [0, 1, 2, 3, 4, 5, -1]],
                    )
                    max_iou = ious.max(-1)[0]
                    if max_iou.mean() < 0.5:
                        count += 1
                        aug_list_batch.append(aug_box3d[:, None, :])
                else:
                    count += 1
                    aug_list_batch.append(aug_box3d[:, None, :])

                if count == 2:
                    break

            if count != 2:
                for _ in range(2 - count):
                    aug_list_batch.append(bbox[:, None, :7])

            aug_list.append(torch.cat(aug_list_batch, 1)[None])

        return torch.cat(aug_list)

    def get_proposal_aware_point_feature(self, src, trajectory_rois, num_rois):
        proposal_aware_polar_point_list = []
        for i in range(trajectory_rois.shape[1]):
            corner_points, _ = get_corner_points_of_roi(
                trajectory_rois[:, i, :, :].contiguous()
            )
            corner_points = corner_points.view(
                self.batch_size, num_rois, -1, corner_points.shape[-1]
            )
            corner_points = corner_points.view(self.batch_size * num_rois, -1)
            trajectory_roi_center = (
                trajectory_rois[:, i, :, :]
                .contiguous()
                .reshape(self.batch_size * num_rois, -1)[:, :3]
            )
            corner_add_center_points = torch.cat(
                [corner_points, trajectory_roi_center], dim=-1
            )
            proposal_aware_car_point = src[
                :, i * self.num_lidar_points : (i + 1) * self.num_lidar_points, :3
            ].repeat(1, 1, 9) - corner_add_center_points.unsqueeze(1).repeat(
                1, self.num_lidar_points, 1
            )

            lwh = (
                trajectory_rois[:, i, :, :]
                .reshape(self.batch_size * num_rois, -1)[:, 3:6]
                .unsqueeze(1)
                .repeat(1, proposal_aware_car_point.shape[1], 1)
            )
            diag_dist = (
                lwh[:, :, 0] ** 2 + lwh[:, :, 1] ** 2 + lwh[:, :, 2] ** 2
            ) ** 0.5
            proposal_aware_polar_point = spherical_coordinate(
                proposal_aware_car_point, diag_dist=diag_dist.unsqueeze(-1)
            )
            proposal_aware_polar_point_list.append(proposal_aware_polar_point)

        proposal_aware_polar_point = torch.cat(proposal_aware_polar_point_list, dim=1)
        proposal_aware_polar_point = torch.cat(
            [proposal_aware_polar_point, src[:, :, 3:]], dim=-1
        )
        proposal_aware_feat = self.up_dimension_geometry(proposal_aware_polar_point)

        return proposal_aware_feat

    def get_trajcetory_point_feature(self, global_trajectory_hypothses, samples):
        candi_length = global_trajectory_hypothses.shape[-2]

        point = crop_current_frame_points(
            self.num_lidar_points, global_trajectory_hypothses, samples["points"]
        )

        point_feat = self.get_proposal_aware_point_feature(
            point.reshape(-1, point.shape[-2], point.shape[-1]),
            global_trajectory_hypothses[:, 0].reshape(self.batch_size, 1, -1, 8),
            self.num_track * candi_length,
        )

        point_feat = point_feat.reshape(-1, self.num_lidar_points, point_feat.shape[-1])

        token = self.token.repeat(
            self.batch_size * self.num_track * self.num_hypo_train, 1, 1
        ).cuda()
        point_feat_list = self.encoder_fg(token, point_feat)

        return point_feat_list

    def get_trajectory_boxes_feature(self, traj_rois):
        traj_boxes = traj_rois.clone()
        batch_size = traj_rois.shape[0]
        num_track, num_candi = (
            traj_rois.shape[2],
            traj_rois.shape[3],
        )
        empty_mask = traj_rois[..., :6].sum(-1) == 0
        traj_boxes[..., 6] = traj_boxes[..., 6] % (2 * np.pi)
        traj_boxes[empty_mask] = 0
        boxes_feat, _ = self.seqboxembed(
            traj_boxes.permute(0, 2, 3, 4, 1)
            .contiguous()
            .view(-1, traj_boxes.shape[-1], traj_boxes.shape[1])
        )
        boxes_feat = boxes_feat.reshape(
            batch_size, num_track, num_candi, boxes_feat.shape[-1]
        )

        return boxes_feat

    def get_trajectory_hypotheses_feat(self, point_feat_list, boxes_feat, pred_labels):
        point_feat = point_feat_list[-1].reshape(
            self.batch_size, self.num_track, self.num_hypo_train, -1
        )
        src = torch.cat(
            [point_feat, boxes_feat, torch.zeros_like(point_feat)[..., :3]], -1
        )

        car_mask = pred_labels == 1
        ped_mask = pred_labels == 2
        cyc_mask = pred_labels == 3

        src[car_mask[:, :, 0]][..., -3:] = self.car_embed
        src[ped_mask[:, :, 0]][..., -3:] = self.ped_embed
        src[cyc_mask[:, :, 0]][..., -3:] = self.cyc_embed

        src = F.relu(self.cls_embed(src))
        return src

    def get_trajectory_hypotheses_feat_inference(
        self, point_feat_list, boxes_feat, pred_labels
    ):
        point_feat = point_feat_list[-1].reshape(
            self.batch_size, self.num_track, -1, self.token.shape[-1]
        )
        src = torch.cat(
            [point_feat, boxes_feat, torch.zeros_like(point_feat)[..., :3]], -1
        )
        self.car_mask = pred_labels == 1
        self.ped_mask = pred_labels == 2
        self.cyc_mask = pred_labels == 3
        src[:, self.car_mask][..., -3:] = self.car_embed
        src[:, self.ped_mask][..., -3:] = self.ped_embed
        src[:, self.cyc_mask][..., -3:] = self.cyc_embed
        src = F.relu(self.cls_embed(src))
        return src

    def organize_proposals(self, pred_boxes3d, pred_scores, pred_labels):
        all_batch_list_boxes = []
        all_batch_list_score = []
        all_batch_list_label = []
        for i in range(len(pred_boxes3d)):
            cur_batch_box = pred_boxes3d[i].reshape(self.traj_length + 1, -1, 9)
            cur_batch_score = pred_scores[i].reshape(self.traj_length + 1, -1)
            cur_batch_label = pred_labels[i].reshape(self.traj_length + 1, -1)
            batch_list = []
            batch_list_score = []
            batch_list_label = []
            for j in range(self.traj_length + 1):
                cur_box = cur_batch_box[j]
                cur_score = cur_batch_score[j]
                cur_label = cur_batch_label[j]
                assert cur_box.shape[0] == cur_score.shape[0]
                mask = self.class_agnostic_nms(
                    cur_box[:, [0, 1, 2, 3, 4, 5, 8]],
                    cur_score.reshape(-1),
                    nms_thresh=self.train_nms_thresh,
                    score_thresh=self.train_score_thresh,
                )
                batch_list.append(cur_box[mask])
                batch_list_score.append(cur_score[mask].reshape(-1, 1))
                batch_list_label.append(cur_label[mask].reshape(-1, 1))

            cur_batch_box, _ = reorder_rois(batch_list)
            all_batch_list_boxes.append(cur_batch_box.reshape(-1, 9))

            cur_batch_score, _ = reorder_rois(batch_list_score)
            all_batch_list_score.append(cur_batch_score.reshape(-1, 1))

            cur_batch_label, _ = reorder_rois(batch_list_label)
            all_batch_list_label.append(cur_batch_label.reshape(-1, 1))

        pred_boxes3d, _ = reorder_rois(all_batch_list_boxes)
        pred_scores, _ = reorder_rois(all_batch_list_score)
        pred_labels, _ = reorder_rois(all_batch_list_label)

        pred_boxes3d_list = pred_boxes3d.reshape(
            self.batch_size, self.traj_length + 1, -1, 9
        )
        det_boxes3d = pred_boxes3d_list[
            :, 0, :, [0, 1, 2, 3, 4, 5, -1]
        ]  # the first is det_boxes
        # det_vel = pred_boxes3d_list[:,0,:,[6,7]]
        pred_vel = pred_boxes3d_list[:, 1:2, :, [6, 7]]

        traj, valid_mask = self.generate_trajectory(
            pred_boxes3d_list[:, 1:, :, [0, 1, 2, 3, 4, 5, 6, 7, 8]]
        )  # the first is det_boxes
        time_sweeps = traj.new_ones(traj.shape[0], traj.shape[1], traj.shape[2], 1)
        for i in range(time_sweeps.shape[1]):
            time_sweeps[:, i] = time_sweeps[:, i] * i * 0.1
        traj = torch.cat([traj[..., [0, 1, 2, 3, 4, 5, 8]], time_sweeps], -1)  # rm vel

        with torch.no_grad():
            pred_hypo = self.get_pred_motion(traj, pred_vel)[:, 0, :, 0]

        traj = traj[:, 1:]  # rm the first t-1 pred boxes
        pred_label_list = pred_labels.reshape(
            self.batch_size, self.traj_length + 1, -1, 1
        )
        pred_labels = pred_label_list[:, 1]

        return pred_hypo, pred_labels, det_boxes3d, traj

    def generate_trajectory_hypothses(
        self, transfered_det, det_boxes3d, traj, num_hypo_det, aug_hypo=None
    ):
        batch_size, num_track = transfered_det.shape[0], transfered_det.shape[1]
        dist = torch.cdist(transfered_det[:, :, :2], det_boxes3d[:, :, :2], 2)
        matched_id = (
            torch.arange(transfered_det.shape[1])
            .cuda()
            .reshape(1, -1, 1)
            .repeat(batch_size, 1, num_hypo_det)
        )
        matched_id[..., :num_hypo_det] = det_boxes3d.shape[1]
        min_value, matched_det_id = torch.topk(-dist, num_hypo_det, -1)
        valid_dist_mask = -min_value < self.dist_thresh
        matched_id[..., :num_hypo_det][valid_dist_mask] = matched_det_id[
            valid_dist_mask
        ]

        batch_index = torch.arange(batch_size).reshape(-1, 1, 1).repeat(1, num_track, 1)
        det_boxes_with_bg = torch.cat(
            [det_boxes3d, torch.zeros(batch_size, 1, 7).cuda()], 1
        )
        group_det_boxes = det_boxes_with_bg[batch_index, matched_id]
        time = torch.zeros([batch_size, num_track, num_hypo_det, 1]).cuda()
        group_det_boxes = torch.cat([group_det_boxes, time], -1)
        transfered_det = transfered_det[:, None, :, None, :]

        if aug_hypo is not None:
            aug_hypo = aug_hypo[:, None, :, :, :]
            time = torch.zeros_like(aug_hypo[..., :1])
            aug_hypo = torch.cat([aug_hypo, time], -1)
            transfered_det = torch.cat([transfered_det, aug_hypo], 3)

        global_candidates = torch.cat([transfered_det, group_det_boxes.unsqueeze(1)], 3)
        traj_repeat = traj.unsqueeze(3).repeat(1, 1, 1, global_candidates.shape[3], 1)
        global_trajectory_hypothses = torch.cat([global_candidates, traj_repeat], 1)

        return global_trajectory_hypothses, global_candidates

    def get_cls_targets(self, pred_boxes3d, global_candidates, targets):
        fg_mask_list = []
        ious_targets = []
        reg_mask_list = []
        gt_boxes_list = []
        batch_size, num_track = (
            pred_boxes3d.shape[0],
            pred_boxes3d.shape[1],
        )
        for i in range(batch_size):
            num_gt = targets[i]["gt_boxes"].shape[0]
            if pred_boxes3d[i].shape[0] > 0 and num_gt > 0:
                rois = global_candidates[i][..., :7].reshape(-1, 7)
                rois_iou = boxes_iou3d_gpu(
                    rois, targets[i]["gt_boxes"][:, [0, 1, 2, 3, 4, 5, -1]].cuda()
                )
                rois_iou = rois_iou.reshape(num_track, self.num_hypo_train, num_gt)
                track_iou = rois_iou[:, 0]
                max_iou, track_id = track_iou.max(-1)
                fg_track_mask = max_iou > 0.5
                reg_mask = rois_iou.max(-1)[0] > 0.5
                group_iou = rois_iou[
                    torch.arange(num_track).cuda(), :, track_id
                ].reshape(-1, self.num_hypo_train)
                group_iou_labels = self.get_iou_labels(group_iou)
                track_id = rois_iou.reshape(-1, num_gt).max(-1)[1]
                ordered_gt_boxes = targets[i]["gt_boxes"][track_id][
                    :, [0, 1, 2, 3, 4, 5, -1]
                ]
                gt_boxes_list.append(ordered_gt_boxes)
                reg_mask_list.append(reg_mask)
                ious_targets.append(group_iou_labels)
                fg_mask_list.append(fg_track_mask)

            else:
                ious_targets.append(
                    torch.zeros([num_track, self.num_hypo_train]).cuda()
                )
                fg_mask_list.append(torch.zeros([num_track]).cuda().bool())
                reg_mask_list.append(
                    torch.zeros([num_track, self.num_hypo_train]).cuda().bool()
                )
                gt_boxes_list.append(
                    torch.zeros([num_track * self.num_hypo_train, 7]).cuda()
                )

        fg_iou_mask = torch.cat(fg_mask_list)
        gt_boxes = torch.cat(gt_boxes_list)
        fg_reg_mask = torch.cat(reg_mask_list)
        fg_reg_mask = (
            fg_reg_mask.reshape(1, -1).repeat(self.num_encoder_layers, 1).reshape(-1)
        )
        ious_targets = (
            torch.cat(ious_targets, 0).reshape(-1).repeat(self.num_encoder_layers)
        )
        return fg_iou_mask, fg_reg_mask, ious_targets, gt_boxes

    def get_reg_targets(self, pred_rois, gt_boxes):
        rois, gt_of_rois = pred_rois[None].clone(), gt_boxes[None].clone()

        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry

        local_rois = rois.clone()
        local_rois[:, :, 0:3] = 0
        local_rois[:, :, 6] = 0

        # transfer LiDAR coords to local coords
        gt_of_rois = rotate_points_along_z(
            points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)
        ).view(1, -1, gt_of_rois.shape[-1])

        # flip orientation if rois have opposite orientation
        heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (
            2 * np.pi
        )  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        flag = heading_label > np.pi
        heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
        heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

        gt_of_rois[:, :, 6] = heading_label
        reg_targets = encode_boxes_res_torch(gt_of_rois[0], local_rois[0])
        reg_targets = reg_targets.repeat(self.num_encoder_layers, 1).reshape(1, -1, 7)
        return reg_targets

    def get_iou_labels(self, cls_iou):
        iou_bg_thresh = 0.25
        iou_fg_thresh = 0.75
        fg_mask = cls_iou > iou_fg_thresh
        bg_mask = cls_iou < iou_bg_thresh
        interval_mask = (fg_mask == 0) & (bg_mask == 0)
        batch_cls_labels = (fg_mask > 0).float()
        batch_cls_labels[interval_mask] = (cls_iou[interval_mask] - iou_bg_thresh) / (
            iou_fg_thresh - iou_bg_thresh
        )
        return batch_cls_labels

    def class_agnostic_nms(
        self,
        pred_boxes3d,
        pred_scores,
        nms_thresh=0.1,
        score_thresh=None,
        nms_pre_maxsize=4096,
        nms_post_maxsize=500,
    ):
        box_preds = pred_boxes3d
        scores = pred_scores
        if score_thresh is not None:
            scores_mask = scores >= score_thresh
            scores = scores[scores_mask]
            box_preds = box_preds[scores_mask]

        rank_scores_nms, indices = torch.topk(
            scores, k=min(nms_pre_maxsize, scores.shape[0])
        )
        box_preds_nms = box_preds[indices][:, :7]

        if box_preds_nms.shape[0] > 0:
            keep_idx, _ = nms_gpu(box_preds_nms, rank_scores_nms, thresh=nms_thresh)

            selected = indices[keep_idx[:nms_post_maxsize]]

            if score_thresh is not None:
                original_idxs = scores_mask.nonzero().view(-1)
                selected = original_idxs[selected]

            return selected
        else:
            return torch.tensor([]).long()

    def generate_trajectory(self, proposals_list):
        cur_batch_boxes = proposals_list[:, 0, :, :]
        trajectory_rois = torch.zeros_like(cur_batch_boxes[:, None, :, :]).repeat(
            1, proposals_list.shape[1], 1, 1
        )
        trajectory_rois[:, 0, :, :] = proposals_list[:, 0, :, :]
        valid_length = torch.zeros(
            [
                trajectory_rois.shape[0],
                trajectory_rois.shape[1],
                trajectory_rois.shape[2],
            ]
        )
        valid_length[:, 0] = 1
        num_frames = proposals_list.shape[1]
        for i in range(1, num_frames):
            frame = torch.zeros_like(cur_batch_boxes)
            frame[:, :, 0:2] = (
                trajectory_rois[:, i - 1, :, 0:2]
                - 0.1 * trajectory_rois[:, i - 1, :, 6:8]
            )
            frame[:, :, 2:] = trajectory_rois[:, i - 1, :, 2:]

            for bs_idx in range(proposals_list.shape[0]):
                iou3d = boxes_iou3d_gpu(
                    frame[bs_idx, :, [0, 1, 2, 3, 4, 5, -1]],
                    proposals_list[bs_idx, i, :, [0, 1, 2, 3, 4, 5, -1]],
                )
                max_overlaps, traj_assignment = torch.max(iou3d, dim=1)
                fg_inds = ((max_overlaps >= 0.5)).nonzero().view(-1)
                valid_length[bs_idx, i, fg_inds] = 1
                trajectory_rois[bs_idx, i, fg_inds, :] = proposals_list[
                    bs_idx, i, traj_assignment[fg_inds]
                ]
        return trajectory_rois, valid_length

    def get_loss(
        self,
        rois,
        gt_boxes,
        point_cls,
        joint_cls,
        boxes_cls,
        point_reg,
        ious_targets,
        reg_targets,
        fg_reg_mask,
        fg_iou_mask,
    ):
        loss_reg = self.reg_loss_func(point_reg, reg_targets)[:, fg_reg_mask]
        loss_reg = loss_reg.sum() / max(fg_reg_mask.sum(), 1)

        loss_corner = get_corner_loss(
            point_reg.reshape(-1, 7),
            rois.repeat(self.num_encoder_layers, 1),
            gt_boxes.repeat(self.num_encoder_layers, 1),
            fg_reg_mask,
        )

        loss_point_cls = F.binary_cross_entropy(
            point_cls.sigmoid().reshape(-1), ious_targets
        )

        index = ious_targets.shape[0] // self.num_encoder_layers
        ious_targets = ious_targets[:index].reshape(
            self.batch_size * self.num_track, self.num_hypo_train
        )
        loss_box_cls = F.binary_cross_entropy(
            boxes_cls.sigmoid()[fg_iou_mask], ious_targets[fg_iou_mask]
        )

        fg_mask_repeat = fg_iou_mask.repeat(self.num_encoder_layers)
        group_ious_repeat = ious_targets.repeat(self.num_encoder_layers, 1)
        loss_joint_cls = F.binary_cross_entropy(
            joint_cls.sigmoid()[fg_mask_repeat], group_ious_repeat[fg_mask_repeat]
        )
        loss_cls = loss_joint_cls + loss_point_cls + loss_box_cls
        loss_reg = loss_reg + loss_corner

        return loss_cls, loss_reg

    def init_trajectory(self, pose, det_boxes, det_scores, det_vels, det_labels):
        self.instances = []
        instance = Instances()
        self.history_trajectory_bank = collections.defaultdict(dict)
        if self.config.model.eval_class == "VEHICLE":
            score_thresh = self.new_born_car
        elif self.config.model.eval_class == "PEDESTRIAN":
            score_thresh = self.new_born_ped
        elif self.config.model.eval_class == "CYCLIST":
            score_thresh = self.new_born_cyc
        else:
            raise NotImplementedError
        score_mask = self.class_agnostic_nms(
            det_boxes[0],
            det_scores[0].reshape(-1),
            nms_thresh=self.nms_thresh,
            score_thresh=score_thresh,
        )
        instance.track_id = torch.arange(score_mask.shape[0]).reshape(-1).cuda()
        instance.pred_boxes = det_boxes[0, score_mask]
        instance.vels = det_vels[0, score_mask]
        instance.scores = det_scores[0, score_mask]
        instance.pred_classes = det_labels[0, score_mask]
        instance.pose = pose
        instance.new_boxes = torch.cat(
            [det_boxes[0, score_mask], det_vels[0, score_mask]], -1
        )
        self.instances.append(instance)
        self.max_id = score_mask.shape[0]

        track_out = {
            "track_scores": det_scores[0, score_mask].detach().cpu(),
            "track_labels": det_labels[0, score_mask].detach().cpu(),
            "track_boxes3d": det_boxes[0, score_mask].detach().cpu(),
            "track_ids": torch.arange(score_mask.shape[0]).reshape(-1).int(),
        }
        global_boxes, global_vels = transform_box_to_global(
            instance.pred_boxes.cpu().numpy(), instance.vels.cpu().numpy(), self.pose
        )

        for index, track_id in enumerate(track_out["track_ids"]):
            track_id = track_id.item()
            self.history_trajectory_bank[track_id]["track_scores"] = []
            self.history_trajectory_bank[track_id]["track_vels"] = []
            self.history_trajectory_bank[track_id]["track_labels"] = []
            self.history_trajectory_bank[track_id]["track_boxes3d"] = []
            self.history_trajectory_bank[track_id]["track_pose"] = []
            self.history_trajectory_bank[track_id]["track_scores"].insert(
                0, instance.scores[index]
            )
            self.history_trajectory_bank[track_id]["track_vels"].insert(
                0, global_vels[index]
            )
            self.history_trajectory_bank[track_id]["track_labels"].insert(
                0, instance.pred_classes[index]
            )
            self.history_trajectory_bank[track_id]["track_boxes3d"].insert(
                0, global_boxes[index]
            )
            self.history_trajectory_bank[track_id]["track_pose"].insert(
                0, instance.pose
            )

        return track_out, instance, global_boxes, global_vels

    def get_point_and_trajectory(
        self, traj, traj_vels, samples, det_boxes, det_vels, det_labels, det_scores
    ):
        pred_traj = self.get_pred_candi(traj, traj_vels)
        pred_candi = pred_traj[:, :, 0, :7]
        cur_vels = self.instances[-1].vels[None]
        det_candi, det_candi_vel, asso_mask = self.get_det_candi(
            self.pose,
            pred_candi,
            cur_vels,
            det_boxes,
            det_vels,
            det_labels,
            det_scores,
            self.frame_id,
        )

        time_sweeps = traj.new_ones(traj.shape[0], traj.shape[1], traj.shape[2], 1)
        for i in range(time_sweeps.shape[1]):
            time_sweeps[:, i] = time_sweeps[:, i] * (i + 1) * 0.1

        traj = torch.cat([traj, time_sweeps], -1)
        (
            trajectory_hypothese,
            global_candidates,
            joint_vels,
        ) = self.genereate_trajcetory_hypotheses_inference(
            pred_traj, det_candi, traj, cur_vels, det_candi_vel
        )
        self.num_candi = trajectory_hypothese.shape[3]
        points = crop_current_frame_points(
            self.num_lidar_points, trajectory_hypothese, samples["points"]
        )

        return points, trajectory_hypothese, global_candidates, joint_vels, asso_mask

    def get_pred_candi(self, traj, traj_vels):
        num_pred = max(1, min(self.num_hypo_inference, traj.shape[1] - 1))
        pred_traj_list = []
        for i in range(num_pred):
            cur_traj = traj[:, i : i + self.history_traj_frames]
            cur_vel = traj_vels[:, i : i + 1]
            pred_traj = self.get_pred_motion(cur_traj, cur_vel)[:, i]
            pred_traj_list.append(pred_traj)

        pred_traj = torch.cat(pred_traj_list, 2)
        empty_mask = pred_traj[..., 3:6].sum(-1) == 0
        pred_traj[empty_mask] = 0

        return pred_traj

    def get_pred_motion(self, traj, pred_vel=None):
        traj_rois = traj.clone().unsqueeze(3)
        batch_size, len_traj, num_track = (
            traj_rois.shape[0],
            traj_rois.shape[1],
            traj_rois.shape[2],
        )
        self.num_future = 10  # pretrained motion model predict 10 future frames
        history_traj = traj_rois
        future_traj_init = traj_rois[:, 0:1].repeat(1, self.num_future, 1, 1, 1)
        future_traj_center = traj_rois[:, 0:1, :, :, :3].repeat(
            1, self.num_future, 1, 1, 1
        )
        pred_vel_hypos = 0.1 * pred_vel.unsqueeze(3).repeat(1, len_traj, 1, 1, 1)

        for i in range(future_traj_center.shape[1]):
            future_traj_center[:, i, :, :, :2] += (
                0.1 * (i + 1) * pred_vel[:, 0].unsqueeze(2)
            )
        future_traj_init[..., :2] = future_traj_center[..., :2]
        empty_mask = (traj_rois[:, 0:1, :, :, 3:6].sum(-1) == 0).repeat(
            1, traj_rois.shape[1], 1, 1
        )

        time_sweeps = torch.ones_like(history_traj)[..., :1]
        for i in range(time_sweeps.shape[1]):
            time_sweeps[:, i] = time_sweeps[:, i] * i * 0.1
        history_traj = torch.cat([history_traj, time_sweeps], -1)
        history_traj_local, history_vel_local = transform_trajs_to_local_coords(
            history_traj,
            center_xyz=history_traj[:, 0:1, :, :, 0:2],
            center_heading=history_traj[:, 0:1, :, :, 6],
            pred_vel_hypo=pred_vel_hypos,
            heading_index=6,
        )

        future_traj_init_local, _ = transform_trajs_to_local_coords(
            future_traj_init,
            center_xyz=history_traj[:, 0:1, :, :, 0:2],
            center_heading=history_traj[:, 0:1, :, :, 6],
            heading_index=6,
        )

        history_traj_local = torch.cat(
            [
                history_traj_local[..., :2],  # xy
                history_traj_local[..., 6:7],  # yaw
                history_vel_local,  # vel
                history_traj_local[..., 7:8],
            ],  # time
            -1,
        )

        history_traj_local = history_traj_local.permute(0, 2, 3, 1, 4).reshape(
            batch_size, num_track * history_traj.shape[3], len_traj, -1
        )
        valid_mask = ~empty_mask.permute(0, 2, 3, 1).reshape(
            batch_size, num_track * history_traj.shape[3], len_traj
        )

        future_traj_pred = self.velboxembed(history_traj_local, valid_mask)
        future_traj_pred = future_traj_pred.reshape(
            batch_size, num_track, history_traj.shape[3], self.num_future, 3
        ).permute(0, 3, 1, 2, 4)
        future_traj_local = future_traj_init_local.clone()
        future_traj_local[..., [0, 1, 6]] = (
            future_traj_pred + future_traj_init_local[..., [0, 1, 6]].detach()
        )

        future_traj = transform_trajs_to_global_coords(
            future_traj_local,
            center_xyz=history_traj[:, 0:1, :, 0:1, 0:2],
            center_heading=history_traj[:, 0:1, :, 0:1, 6],
            heading_index=6,
        )[0]

        return future_traj

    def get_det_candi(
        self,
        pose,
        transfered_det,
        cur_vels,
        det_boxes,
        det_vels,
        det_labels,
        det_scores,
        frame_id,
    ):
        time_lag = 0.1
        global_boxes, global_vels = transform_box_to_global(
            det_boxes[0].cpu(), det_vels[0].cpu(), pose
        )
        current_det = []
        for i in range(det_boxes.shape[1]):
            current_det.append(
                {
                    "translation": global_boxes[i, :2].cpu().numpy(),
                    "velocity": global_vels[i].cpu().numpy(),
                    "detection_name": self.WAYMO_TRACKING_NAMES[
                        int(det_labels[0, i] - 1)
                    ],
                    "score": det_scores[0, i].cpu().numpy(),
                    "box_id": i,
                    "label_preds": det_labels[0, i].cpu().numpy(),
                }
            )

        outputs = self.tracker.step_centertrack(current_det, time_lag, frame_id)
        tracking_ids = []
        box_ids = []
        for item in outputs:
            if item["active"] == 0:
                continue
            box_ids.append(item["box_id"])
            tracking_ids.append(item["tracking_id"])

        remained_box_ids = np.array(box_ids)
        det_candi = torch.zeros_like(transfered_det)
        det_candi_vel = torch.zeros_like(cur_vels)
        asso_mask = torch.zeros(transfered_det.shape[1]).bool()
        for i in range(remained_box_ids.shape[0]):
            track_id = tracking_ids[i]
            det_candi[0][track_id] = det_boxes[0][remained_box_ids][i]
            det_candi_vel[0][track_id] = det_vels[0][remained_box_ids][i]
            asso_mask[track_id] = True

        # time = torch.zeros_like(det_candi)[...,:1]
        # det_candi = torch.cat([det_candi,time],-1)

        return det_candi, det_candi_vel, asso_mask

    def genereate_trajcetory_hypotheses_inference(
        self, pred_hypo, cp_matched_boxes, traj, cur_vels, det_candi_vel
    ):
        time = torch.zeros_like(pred_hypo)[..., :1]
        pred_hypo = torch.cat([pred_hypo, time], -1).unsqueeze(1)
        group_det_boxes = torch.cat(
            [
                cp_matched_boxes.unsqueeze(2),
                torch.zeros([self.batch_size, self.num_track, 1, 1]).cuda(),
            ],
            -1,
        )

        global_candidates = torch.cat([pred_hypo, group_det_boxes.unsqueeze(1)], 3)
        trajectory_hypotheses = torch.cat(
            [
                global_candidates,
                traj.unsqueeze(3).repeat(1, 1, 1, global_candidates.shape[3], 1),
            ],
            1,
        )
        vels_hypotheses = cur_vels[:, :, None, :].repeat(
            1, 1, global_candidates.shape[3] - 1, 1
        )
        vels_hypotheses = torch.cat([vels_hypotheses, det_candi_vel.unsqueeze(2)], 2)
        vels_hypotheses = vels_hypotheses.reshape(self.num_track, -1, 2)
        global_candidates = global_candidates.reshape(self.num_track, -1, 8)

        return trajectory_hypotheses, global_candidates, vels_hypotheses

    def generate_refined_boxes(self, rois, box_preds=None):
        code_size = rois.shape[-1]
        num_rois = rois.shape[0]
        roi_ry = rois[:, :, 6].view(-1)
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0
        batch_box_preds = decode_torch(box_preds, local_rois).view(-1, code_size)

        batch_box_preds = rotate_points_along_z(
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)

        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view(num_rois, -1, code_size)
        batch_box_preds = torch.cat([batch_box_preds, rois[:, :, 7:]], -1)
        return batch_box_preds

    def get_keep_mask(self, fg_confidence, asso_mask):
        keep_mask = torch.zeros_like(fg_confidence[:, 0]).bool()
        keep_mask[asso_mask] = True
        track_score_mask = torch.zeros_like(keep_mask)
        track_score_mask[self.car_mask] = (
            fg_confidence[:, 0].reshape(-1)[self.car_mask] > self.keep_thresh_car
        )
        track_score_mask[self.ped_mask] = (
            fg_confidence[:, 0].reshape(-1)[self.ped_mask] > self.keep_thresh_ped
        )
        track_score_mask[self.cyc_mask] = (
            fg_confidence[:, 0].reshape(-1)[self.cyc_mask] > self.keep_thresh_cyc
        )
        keep_mask[~asso_mask] = track_score_mask[~asso_mask]
        return keep_mask

    def update_trajectory(self, output_new, track_new):
        boxes_new = output_new["pred_boxes"].reshape(1, -1, 7)
        scores_new = output_new["pred_logits"].reshape(1, -1)
        vels_new = output_new["pred_vels"].reshape(1, -1, 2)
        labels_new = output_new["pred_labels"]
        matched_boxes = track_new["matched_boxes"]
        refined_matched_boxes = track_new["refined_matched_boxes"]
        track_id = track_new["track_id"]
        matched_vels = track_new["matched_vels"]
        matched_scores = track_new["matched_scores"]
        matched_labels = track_new["matched_labels"]

        if (
            self.frame_id > 0
            and boxes_new[0].shape[0] > 0
            and matched_boxes.shape[0] > 0
        ):
            ious_det2track = boxes_iou3d_gpu(boxes_new[0], matched_boxes)
            mask = ious_det2track.max(-1)[0] > self.new_born_nms_thresh
            scores_new[0][mask] = 0

        scores_new_mask = torch.zeros_like(scores_new).bool()
        new_car_mask = labels_new == 1
        new_ped_mask = labels_new == 2
        new_cyc_mask = labels_new == 3
        scores_new_mask[new_car_mask] = scores_new[new_car_mask] > self.new_born_car
        scores_new_mask[new_ped_mask] = scores_new[new_ped_mask] > self.new_born_ped
        scores_new_mask[new_cyc_mask] = scores_new[new_cyc_mask] > self.new_born_cyc

        if scores_new_mask.sum() > 0:
            new_det_scores_mask = scores_new_mask[0]
            new_det_boxes = boxes_new[0, new_det_scores_mask]
            new_det_scores = scores_new[0, new_det_scores_mask].reshape(-1)
            new_det_vels = vels_new[0, new_det_scores_mask]
            new_det_labels = labels_new[0, new_det_scores_mask]
            new_track_id = self.max_id + 1 + torch.arange(new_det_boxes.shape[0]).cuda()
            self.max_id = self.max_id + 1 + new_det_boxes.shape[0]

        else:
            new_det_scores_mask = []
            new_det_boxes = torch.tensor([])
            new_det_scores = torch.tensor([])
            new_det_vels = torch.tensor([])
            new_det_labels = torch.tensor([])
            new_track_id = torch.tensor([])

        instance = Instances()
        instance.track_id = torch.cat([track_id.cuda(), new_track_id.cuda()], 0)
        instance.pred_boxes = torch.cat([matched_boxes.cuda(), new_det_boxes.cuda()], 0)
        instance.refined_pred_boxes = torch.cat(
            [refined_matched_boxes.cuda(), new_det_boxes.cuda()], 0
        )
        instance.new_boxes = boxes_new[0, new_det_scores_mask]
        instance.scores = torch.cat([matched_scores.cuda(), new_det_scores.cuda()], 0)
        instance.vels = torch.cat([matched_vels.cuda(), new_det_vels.cuda()], 0)
        instance.pred_classes = torch.cat(
            [matched_labels.cuda(), new_det_labels.cuda()], 0
        )
        instance.pose = self.pose
        self.instances.append(instance)

        track_out = {
            "track_scores": instance.scores.cpu(),
            "track_labels": instance.pred_classes.cpu(),
            "track_boxes3d": instance.refined_pred_boxes.cpu(),
            "track_ids": instance.track_id.detach().cpu().int(),
        }
        global_boxes, global_vels = transform_box_to_global(
            instance.pred_boxes.cpu().numpy(), instance.vels.cpu().numpy(), self.pose
        )

        for index, track_id in enumerate(track_out["track_ids"]):
            track_id = track_id.item()
            if track_id not in self.history_trajectory_bank.keys():
                self.history_trajectory_bank[track_id]["track_scores"] = []
                self.history_trajectory_bank[track_id]["track_vels"] = []
                self.history_trajectory_bank[track_id]["track_labels"] = []
                self.history_trajectory_bank[track_id]["track_boxes3d"] = []
                self.history_trajectory_bank[track_id]["track_pose"] = []

            self.history_trajectory_bank[track_id]["track_scores"].insert(
                0, instance.scores[index]
            )
            self.history_trajectory_bank[track_id]["track_vels"].insert(
                0, global_vels[index]
            )
            self.history_trajectory_bank[track_id]["track_labels"].insert(
                0, instance.pred_classes[index]
            )
            self.history_trajectory_bank[track_id]["track_boxes3d"].insert(
                0, global_boxes[index]
            )
            self.history_trajectory_bank[track_id]["track_pose"].insert(
                0, instance.pose
            )

        self.update_global_hypotheses_for_dist_asso(global_boxes, global_vels, instance)
        return track_out

    def update_global_hypotheses_for_dist_asso(
        self, global_boxes, global_vels, instance
    ):
        tracks = []
        for i in range(instance.pred_boxes.shape[0]):
            tracks.append(
                {
                    "translation": global_boxes[i, :2].cpu().numpy(),
                    "ct": global_boxes[i, :2].cpu().numpy(),
                    "velocity": global_vels[i].cpu().numpy(),
                    "detection_name": self.WAYMO_TRACKING_NAMES[
                        int(instance.pred_classes[i] - 1)
                    ],
                    "score": instance.scores[i].cpu().numpy(),
                    "box_id": i,
                    "tracking_id": i,
                    "label_preds": instance.pred_classes[i].cpu().numpy(),
                    "active": 1,
                    "age": 1,
                }
            )
        self.tracker.reset(self.max_id, tracks)


def collate(batch_list, device):
    targets_merged = collections.defaultdict(list)
    for targets in batch_list:
        for target in targets:
            for k, v in target.items():
                targets_merged[k].append(v)
    batch_size = len(batch_list)
    ret = {}
    for key, elems in targets_merged.items():
        if key in ["voxels", "num_points_per_voxel", "num_voxels"]:
            ret[key] = torch.tensor(np.concatenate(elems, axis=0)).to(device)
        elif key in [
            "gt_boxes",
            "labels",
            "gt_names",
            "difficulty",
            "num_points_in_gt",
        ]:
            max_gt = -1
            for k in range(batch_size):
                max_gt = max(max_gt, len(elems[k]))
                batch_gt_boxes3d = np.zeros(
                    (batch_size, max_gt, *elems[0].shape[1:]), dtype=elems[0].dtype
                )
            for i in range(batch_size):
                batch_gt_boxes3d[i, : len(elems[i])] = elems[i]
            if key != "gt_names":
                batch_gt_boxes3d = torch.tensor(batch_gt_boxes3d, device=device)
            ret[key] = batch_gt_boxes3d
        elif key == "calib":
            ret[key] = {}
            for elem in elems:
                for k1, v1 in elem.items():
                    if k1 not in ret[key]:
                        ret[key][k1] = [v1]
                    else:
                        ret[key][k1].append(v1)
            for k1, v1 in ret[key].items():
                ret[key][k1] = torch.tensor(np.stack(v1, axis=0))
        elif key in ["coordinates", "points"]:
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode="constant", constant_values=i
                )
                coors.append(coor_pad)
            ret[key] = torch.tensor(np.concatenate(coors, axis=0)).to(device)
        else:
            ret[key] = np.stack(elems, axis=0)

    return ret
