import collections
import numpy as np
import torch
from torch import nn
from efg.modeling.operators import nms_gpu, boxes_iou3d_gpu 
import torch.nn.functional as F
from  modules.utils import transform_trajs_to_local_coords
from pointnet import  MotionEncoder
from losses import WeightedSmoothL1Loss

class MotionPrediction(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = torch.device(config.model.device)
        self.config = config
        self.is_train = config.task == 'train'
        self.num_hypo = config.model.num_hypo
        hidden_dim = config.model.hidden_dim
        input_dim = config.model.motion_input_dim
        self.num_future = config.dataset.future_frames
        self.velboxembed = MotionEncoder(input_dim,hidden_dim,out_channels=self.num_future*3).to(self.device)
        self.traj_length = config.dataset.traj_length
        self.dist_thresh = config.model.dist_thresh
        self.reg_loss_func = WeightedSmoothL1Loss(code_weights=None)

    def genereate_trajcetory_hypotheses(self,transfered_det,det_boxes3d,traj,num_hypo):

        batch_size, num_track = transfered_det.shape[0],transfered_det.shape[1]
        dist = torch.cdist(transfered_det[:,:,:2], det_boxes3d[:,:,:2],2)
        matched_id  = torch.arange(transfered_det.shape[1]).cuda().reshape(1,-1,1).repeat(batch_size,1,num_hypo-1)
        matched_id[...,:num_hypo-1] = det_boxes3d.shape[1]
        min_value,matched_det_id = torch.topk(-dist,num_hypo-1,-1)
        valid_dist_mask = -min_value < self.dist_thresh
        matched_id[...,:num_hypo-1][valid_dist_mask] = matched_det_id[valid_dist_mask]
        batch_index = torch.arange(batch_size).reshape(-1,1,1).repeat(1,num_track,1)
        det_boxes_with_bg = torch.cat([det_boxes3d,torch.zeros(batch_size,1,7).cuda()],1)
        group_det_boxes = det_boxes_with_bg[batch_index,matched_id]
        time = torch.zeros([batch_size,num_track,num_hypo-1,1]).cuda()
        group_det_boxes = torch.cat([group_det_boxes,time],-1)
        transfered_det = transfered_det[:,None,:,None,:]
        global_candidates = torch.cat([transfered_det,group_det_boxes.unsqueeze(1)],3)
        traj_repeat = traj.unsqueeze(3).repeat(1,1,1,global_candidates.shape[3],1)
        trajcetory_hypotheses = torch.cat([global_candidates, traj_repeat],1)

        return trajcetory_hypotheses, global_candidates, valid_dist_mask

    def get_pred_traj(self,traj_rois, valid_mask=None, pred_vel=None, det_vel=None, pred_label=None):

        batch_size, len_traj, num_track, num_hypo = traj_rois.shape[0], traj_rois.shape[1], traj_rois.shape[2], traj_rois.shape[3]

        history_traj = traj_rois.clone()
        future_traj_init = traj_rois.clone()[:,0:1].repeat(1,self.num_future,1,1,1)
        future_traj_center = traj_rois.clone()[:,0:1,:,:,:3].repeat(1,self.num_future,1,1,1)
        pred_vel_hypos = 0.1 * pred_vel.unsqueeze(1).unsqueeze(3).repeat(1,len_traj,1,2,1)
        pred_vel_hypos[:,:,:,1] =  0.1*det_vel[:,None,:]
        for i in range(future_traj_center.shape[1]):
            future_traj_center[:,i,:,:,:2] += 0.1*(i+1)*pred_vel.unsqueeze(2)
        future_traj_init[...,:2] = future_traj_center[...,:2]
        empty_mask = (traj_rois[:,0:1,:,:,3:6].sum(-1) ==0).repeat(1,traj_rois.shape[1],1,1)
        history_traj_local, history_vel_local = transform_trajs_to_local_coords(history_traj,
                                                                         center_xyz=history_traj[:, 0:1,:,:,0:2],
                                                                         center_heading=history_traj[:, 0:1,:,:, 6],
                                                                         pred_vel_hypo=pred_vel_hypos,
                                                                         heading_index=6)

        future_traj_init_local, _ = transform_trajs_to_local_coords(future_traj_init,
                                                                      center_xyz=history_traj[:, 0:1,:,:,0:2],
                                                                      center_heading=history_traj[:, 0:1, :,:, 6],
                                                                      heading_index=6)

        history_traj_local = torch.cat([history_traj_local[...,:2],  # xy
                                       history_traj_local[...,6:7], # lw
                                       history_vel_local,  # vel
                                       history_traj_local[...,7:8]], # time
                                       -1)

        history_traj_local = history_traj_local.permute(0,2,3,1,4).reshape(batch_size,num_track*history_traj.shape[3],len_traj,-1)
        valid_mask = ~empty_mask.permute(0,2,3,1).reshape(batch_size,num_track*history_traj.shape[3],len_traj)

        future_traj_pred = self.velboxembed(history_traj_local,valid_mask)
        future_traj_pred = future_traj_pred.reshape(batch_size,num_track, history_traj.shape[3],self.num_future,3).permute(0,3,1,2,4)
        future_traj = future_traj_init_local.clone()
        future_traj[...,[0,1,6]] = future_traj_pred + future_traj_init_local[...,[0,1,6]].detach()

        return future_traj

    def organize_proposals(self,batch_size,pred_boxes3d,pred_scores,pred_labels):

        all_batch_list_boxes = []
        all_batch_list_score = []
        all_batch_list_label = []
        for i in range(len(pred_boxes3d)):
            cur_batch_box = pred_boxes3d[i].reshape(self.traj_length+1,-1,9)
            cur_batch_score = pred_scores[i].reshape(self.traj_length+1,-1)
            cur_batch_label = pred_labels[i].reshape(self.traj_length+1,-1)
            batch_list = []
            batch_list_score =  []
            batch_list_label = []
            for j in range(self.traj_length+1):
                cur_box = cur_batch_box[j]
                cur_score = cur_batch_score[j]
                cur_label = cur_batch_label[j]
                assert cur_box.shape[0] == cur_score.shape[0]
                mask = self.class_agnostic_nms(cur_box[:,[0,1,2,3,4,5,8]],
                                                cur_score.reshape(-1),
                                                nms_thresh=self.config.dataset.nms_thresh,
                                                score_thresh=self.config.dataset.score_thresh)
                batch_list.append(cur_box[mask])
                batch_list_score.append(cur_score[mask].reshape(-1,1))
                batch_list_label.append(cur_label[mask].reshape(-1,1))

            cur_batch_box,_ = self.reorder_rois(batch_list)
            all_batch_list_boxes.append(cur_batch_box.reshape(-1,9))

            cur_batch_score,_ = self.reorder_rois(batch_list_score)
            all_batch_list_score.append(cur_batch_score.reshape(-1,1))

            cur_batch_label,_ = self.reorder_rois(batch_list_label)
            all_batch_list_label.append(cur_batch_label.reshape(-1,1))

        pred_boxes3d, _ = self.reorder_rois(all_batch_list_boxes)
        pred_scores, _ = self.reorder_rois(all_batch_list_score)
        pred_labels, _ = self.reorder_rois(all_batch_list_label)

        pred_boxes3d_list = pred_boxes3d.reshape(batch_size,self.traj_length+1,-1,9) 
        det_boxes3d = pred_boxes3d_list[:,0,:,[0,1,2,3,4,5,-1]] # the first is det_boxes
        pred_boxes3d = pred_boxes3d_list[:,1,:,[0,1,2,3,4,5,-1]]

        traj, valid_mask = self.generate_trajectory(pred_boxes3d_list[:,1:,:,[0,1,2,3,4,5,6,7,8]]) # the first is det_boxes
        time_sweeps = traj.new_ones(traj.shape[0],traj.shape[1],traj.shape[2],1)
        for i in range(time_sweeps.shape[1]):
            time_sweeps[:,i] = time_sweeps[:,i] * i * 0.1
        traj = torch.cat([traj[...,[0,1,2,3,4,5,8]],time_sweeps],-1) #rm vel
        pred_boxes3d = traj[:,0] # t-1 pred boxes
        traj = traj[:,1:]       #rm the first t-1 pred boxes

        det_vel = pred_boxes3d_list[:,0,:,[6,7]]
        pred_vel = pred_boxes3d_list[:,1,:,[6,7]]

        pred_score_list = pred_scores.reshape(batch_size,self.traj_length+1,-1,1) 
        pred_scores = pred_score_list[:,1]
        pred_label_list = pred_labels.reshape(batch_size,self.traj_length+1,-1,1) 
        pred_labels = pred_label_list[:,1]

        return pred_boxes3d, pred_scores, pred_labels, pred_vel, det_boxes3d, det_vel, traj,valid_mask

    def get_targets(self,pred_boxes3d,targets,global_candidates):

        reg_mask_list = []
        gt_boxes_list = []
        gt_future_list = []
        gt_future_list_local = []
        batch_size = pred_boxes3d.shape[0]
        num_track = global_candidates.shape[2]
        for i in range(batch_size):
            if pred_boxes3d[i].shape[0] > 0 and targets[i]["gt_boxes"].shape[0] > 0:
                rois = global_candidates[i][...,:7].reshape(-1,7)
                track_iou = boxes_iou3d_gpu(rois, targets[i]["gt_boxes"][:,[0,1,2,3,4,5,-1]].cuda())
                max_iou, track_id = track_iou.max(-1)
                reg_mask = max_iou > 0.5
                ordered_gt_boxes = targets[i]['gt_boxes'][track_id][:,[0,1,2,3,4,5,-1]]
                pred_gt_boxes = targets[i]['future_gt_boxes'].reshape(-1,targets[i]['gt_boxes'].shape[0],9)
                track_id_vel = track_id.reshape(-1,global_candidates.shape[3])[:,0].reshape(-1)
                ordered_future_gt_boxes = pred_gt_boxes[:,track_id_vel][None, :,:,[0,1,2,3,4,5,-1]].unsqueeze(3)
                ordered_future_gt_boxes_local = transform_trajs_to_local_coords(ordered_future_gt_boxes,
                                                                                center_xyz=ordered_future_gt_boxes[:,0:1,:,0:1,:3],
                                                                                center_heading=ordered_future_gt_boxes[:,0:1,:,0:1,6],
                                                                                heading_index=6)[0].squeeze(3)
                gt_boxes_list.append(ordered_gt_boxes)
                gt_future_list_local.append(ordered_future_gt_boxes_local[:,1:])
                gt_future_list.append(ordered_future_gt_boxes[:,1:])
                reg_mask_list.append(reg_mask)
            else:
                pad = global_candidates[i][...,:7].reshape(-1,7)
                reg_mask_list.append(torch.zeros([pad.shape[0]]).cuda().bool())
                gt_boxes_list.append(torch.zeros([pad.shape[0],7]).cuda())
                gt_future_list.append(torch.zeros([1,self.num_future,num_track,1,7]).cuda())
                gt_future_list_local.append(torch.zeros([1,self.num_future,num_track,7]).cuda())

        gt_boxes = torch.cat(gt_boxes_list)
        gt_future_boxes = torch.cat(gt_future_list) 
        gt_future_traj_local = torch.cat(gt_future_list_local)
        gt_future_traj_local = gt_future_traj_local[:,:,:,None,[0,1,6]] # [0,1,6] means x,y and yaw
        fg_reg_mask = torch.cat(reg_mask_list)
        fg_reg_mask = fg_reg_mask.reshape(batch_size,1,num_track,self.num_hypo)
        return gt_boxes, gt_future_boxes, gt_future_traj_local, fg_reg_mask

    def class_agnostic_nms(self, pred_boxes3d,pred_scores, nms_thresh=0.1, score_thresh=None,
                        nms_pre_maxsize=4096, nms_post_maxsize=500):

        box_preds = pred_boxes3d
        scores = pred_scores
        if score_thresh is not None:
            scores_mask = (scores >= score_thresh)
            scores = scores[scores_mask]
            box_preds = box_preds[scores_mask]
        rank_scores_nms, indices = torch.topk(
            scores, k=min(nms_pre_maxsize, scores.shape[0])
        )
        box_preds_nms = box_preds[indices][:,:7]
        if box_preds_nms.shape[0] >0:
            keep_idx, _ = nms_gpu(
                box_preds_nms, rank_scores_nms, thresh=nms_thresh
            )
            selected = indices[keep_idx[:nms_post_maxsize]]
            if score_thresh is not None:
                original_idxs = scores_mask.nonzero().view(-1)
                selected = original_idxs[selected]

            return selected
        else:
            return torch.tensor([]).long()

    @staticmethod
    def reorder_rois(pred_bboxes):

        num_max_rois = max([len(bbox) for bbox in pred_bboxes])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        ordered_bboxes = torch.zeros([len(pred_bboxes),num_max_rois,pred_bboxes[0].shape[-1]]).cuda()
        valid_mask = np.zeros([len(pred_bboxes),num_max_rois,pred_bboxes[0].shape[-1]])
        for bs_idx in range(ordered_bboxes.shape[0]):
            ordered_bboxes[bs_idx,:len(pred_bboxes[bs_idx])] = pred_bboxes[bs_idx]
            valid_mask[bs_idx,:len(pred_bboxes[bs_idx])] = 1
        return ordered_bboxes, torch.from_numpy(valid_mask).cuda().bool()

    def generate_trajectory(self,proposals_list):

        cur_batch_boxes = proposals_list[:,0,:,:]
        trajectory_rois = torch.zeros_like(cur_batch_boxes[:,None,:,:]).repeat(1,proposals_list.shape[1],1,1)
        trajectory_rois[:,0,:,:]= proposals_list[:,0,:,:] 
        valid_length = torch.zeros([trajectory_rois.shape[0],trajectory_rois.shape[1],trajectory_rois.shape[2]])
        valid_length[:,0] = 1
        num_frames = proposals_list.shape[1]
        for i in range(1,num_frames):
            frame = torch.zeros_like(cur_batch_boxes)
            frame[:,:,0:2] = trajectory_rois[:,i-1,:,0:2] - 0.1*trajectory_rois[:,i-1,:,6:8]
            frame[:,:,2:] = trajectory_rois[:,i-1,:,2:]

            for bs_idx in range(proposals_list.shape[0]):
                iou3d = boxes_iou3d_gpu(frame[bs_idx,:,[0,1,2,3,4,5,-1]], proposals_list[bs_idx,i,:,[0,1,2,3,4,5,-1]])
                max_overlaps, traj_assignment = torch.max(iou3d, dim=1)
                fg_inds = ((max_overlaps >= 0.5)).nonzero().view(-1)
                valid_length[bs_idx,i,fg_inds] = 1
                trajectory_rois[bs_idx,i,fg_inds,:] = proposals_list[bs_idx,i,traj_assignment[fg_inds]]

        return trajectory_rois ,valid_length

    def forward(self, batched_inputs):

        batch_size = len(batched_inputs)
        targets = [bi[1]["annotations"] for bi in batched_inputs]
        for key in ['gt_boxes','future_gt_boxes', 'difficulty', 'num_points_in_gt', 'labels']:
            for i in range(batch_size):
                targets[i][key] = torch.tensor(targets[i][key], device=self.device)
        pred_boxes3d = [torch.from_numpy(bi[1]["annotations"]['pred_boxes3d']).cuda() for bi in batched_inputs]
        pred_scores = [torch.from_numpy(bi[1]["annotations"]['pred_scores']).cuda() for bi in batched_inputs]
        pred_labels = [torch.from_numpy(bi[1]["annotations"]['pred_labels']).cuda() for bi in batched_inputs]

        outputs = self.organize_proposals(batch_size,pred_boxes3d,pred_scores,pred_labels)
        pred_boxes3d, pred_scores, pred_labels, pred_vel, det_boxes3d, det_vel, traj, valid_mask = outputs
        num_track = pred_boxes3d.shape[1]
        batch_size = pred_boxes3d.shape[0]
        if num_track >0 and det_boxes3d.shape[1] > 0:
            loss_dict = {}
            trajectory_hypothese, global_candidates, _ = \
            self.genereate_trajcetory_hypotheses(pred_boxes3d,det_boxes3d,traj,self.num_hypo)
            future_traj_local = self.get_pred_traj(trajectory_hypothese,valid_mask,pred_vel,det_vel)
            gt_boxes, gt_future_boxes, gt_future_traj_local, fg_reg_mask = self.get_targets(pred_boxes3d,targets,global_candidates)
            valid_gt_mask = (gt_future_boxes[...,3:6].sum(-1) > 0).repeat(1,1,1,self.num_hypo)
            valid_fg_mask = fg_reg_mask.repeat(1,self.num_future,1,1)
            valid_mask = torch.logical_and(valid_gt_mask,valid_fg_mask)
            loss = self.reg_loss_func(future_traj_local[...,[0,1,6]], gt_future_traj_local.repeat(1,1,1,self.num_hypo,1))[valid_mask]
            loss = loss.sum()/valid_mask.sum()
            if gt_boxes.shape[0] > 0:
                loss_dict.update({
                            'loss': loss,
                            })
            else:
                loss_dict = {'loss':torch.tensor([0.0]).cuda().reshape(1,-1)}
        else:
            loss_dict = {'loss':torch.tensor([0.0]).cuda().reshape(1,-1)}

        return loss_dict

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
        elif key in ["gt_boxes", "labels", "gt_names", "difficulty", "num_points_in_gt"]:
            max_gt = -1
            for k in range(batch_size):
                max_gt = max(max_gt, len(elems[k]))
                batch_gt_boxes3d = np.zeros(
                    (batch_size, max_gt, *elems[0].shape[1:]),
                    dtype=elems[0].dtype)
            for i in range(batch_size):
                batch_gt_boxes3d[i, :len(elems[i])] = elems[i]
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
                coor_pad = np.pad(coor, ((0, 0), (1, 0)),
                                  mode="constant",
                                  constant_values=i)
                coors.append(coor_pad)
            ret[key] = torch.tensor(np.concatenate(coors, axis=0)).to(device)
        else:
            ret[key] = np.stack(elems, axis=0)

    return ret

