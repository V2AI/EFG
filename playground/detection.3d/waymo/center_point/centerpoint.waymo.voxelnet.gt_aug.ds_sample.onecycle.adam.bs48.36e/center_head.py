import copy
import logging

import numpy as np

import torch
from torch import nn

from efg.modeling.common import get_norm, weight_init
from efg.modeling.utils import Sequential

import box_torch_ops
from centernet_loss import FastFocalLoss, RegLoss
from circle_nms_jit import circle_nms

logger = logging.getLogger(__name__)


class SepHead(nn.Module):
    def __init__(self, in_channels, heads, head_conv=64, final_kernel=1, bn=None, init_bias=-2.19):
        super().__init__()

        self.heads = heads
        for head in self.heads:
            classes, num_conv = self.heads[head]
            fc = Sequential()
            for i in range(num_conv - 1):
                fc.add(
                    nn.Conv2d(
                        in_channels, head_conv, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True
                    )
                )
                if bn is not None:
                    fc.add(get_norm(bn, head_conv))
                fc.add(nn.ReLU())
            fc.add(
                nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True)
            )
            if "hm" in head:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        weight_init.kaiming_init(m)

            self.__setattr__(head, fc)

    def forward(self, x):
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)
        return ret_dict


class CenterHead(nn.Module):
    def __init__(self, config, logger=None, init_bias=-2.19, share_conv_channel=64, num_hm_conv=2):
        super(CenterHead, self).__init__()

        tasks = config.model.head.tasks
        num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.code_weights = config.model.head.misc.code_weights
        self.weight = config.model.head.misc.weight  # weight between hm loss and loc loss
        self.dataset = config.model.head.misc.dataset

        self.common_heads = config.model.head.misc.common_heads

        self.in_channels = config.model.head.in_channels
        self.num_classes = num_classes

        self.criterion = FastFocalLoss()
        self.criterion_reg = RegLoss()

        self.box_n_dim = 9 if "vel" in config.model.head.misc.common_heads else 7
        self.use_direction_classifier = False

        # Configurable BN
        self._norm_config = config.model.neck.norm

        if not logger:
            logger = logging.getLogger("CenterHead")
        self.logger = logger
        logger.info(f"num_classes: {num_classes}")

        # a shared convolution
        self.shared_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, share_conv_channel, kernel_size=3, padding=1, bias=True),
            get_norm(self._norm_config, share_conv_channel),
            nn.ReLU(inplace=True),
        )

        self.tasks = nn.ModuleList()
        logger.info(f"Use HM Bias: {init_bias}")

        for num_cls in num_classes:
            heads = copy.deepcopy(self.common_heads)
            heads.update(dict(hm=(num_cls, num_hm_conv)))
            self.tasks.append(
                SepHead(share_conv_channel, heads, bn=self._norm_config, init_bias=init_bias, final_kernel=3)
            )

        logger.info("Finish CenterHead Initialization")

    def forward(self, x):
        ret_dicts = []
        x = self.shared_conv(x)
        for task in self.tasks:
            ret_dicts.append(task(x))
        return ret_dicts

    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
        return y

    def loss(self, example, preds_dicts):
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict["hm"] = self._sigmoid(preds_dict["hm"])
            hm_loss = self.criterion(
                preds_dict["hm"],
                example["hm"][task_id],
                example["ind"][task_id],
                example["mask"][task_id],
                example["cat"][task_id],
            )
            target_box = example["anno_box"][task_id]
            # reconstruct the anno_box from multiple reg heads
            if "vel" in preds_dict:
                preds_dict["anno_box"] = torch.cat(
                    (preds_dict["reg"], preds_dict["height"], preds_dict["dim"], preds_dict["vel"], preds_dict["rot"]),
                    dim=1,
                )
            else:
                preds_dict["anno_box"] = torch.cat(
                    (preds_dict["reg"], preds_dict["height"], preds_dict["dim"], preds_dict["rot"]), dim=1
                )
                target_box = target_box[..., [0, 1, 2, 3, 4, 5, -2, -1]]  # remove vel target

            ret = {}
            # Regression loss for dimension, offset, height, rotation
            box_loss = self.criterion_reg(
                preds_dict["anno_box"],
                example["mask"][task_id],
                example["ind"][task_id],
                target_box,
            )

            loc_loss = (box_loss * box_loss.new_tensor(self.code_weights)).sum()

            loss = hm_loss + self.weight * loc_loss
            ret.update(
                {
                    "loss": loss,
                    "hm_loss": hm_loss.detach(),
                    "loc_loss": loc_loss,
                    # 'loc_loss_elem': box_loss.detach(),
                    "num_positive": example["mask"][task_id].float().sum(),
                }
            )
            rets.append(ret)

        """convert batch-key to key-batch
        """
        all_rets = {}
        idx = 0
        for ret in rets:
            for k, v in ret.items():
                all_rets[str(idx) + "_" + k] = v
            idx += 1
        return all_rets

    @torch.no_grad()
    def predict(self, example, preds_dicts, test_config):
        """decode, nms, then return the detection result. Additionaly support double flip testing"""
        # get loss info
        rets = []
        metas = []
        double_flip = test_config.get("double_flip", False)
        post_center_range = test_config.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=preds_dicts[0]["hm"].dtype,
                device=preds_dicts[0]["hm"].device,
            )
        for task_id, preds_dict in enumerate(preds_dicts):
            # convert N C H W to N H W C
            for key, val in preds_dict.items():
                preds_dict[key] = val.permute(0, 2, 3, 1).contiguous()
            batch_size = preds_dict["hm"].shape[0]
            if double_flip:
                assert batch_size % 4 == 0, print(batch_size)
                batch_size = int(batch_size / 4)
                for k in preds_dict.keys():
                    # transform the prediction map back to their original coordinate befor flipping
                    # the flipped predictions are ordered in a group of 4. The first one is the original pointcloud
                    # the second one is X flip pointcloud(y=-y), the third one is Y flip pointcloud(x=-x),
                    # and the last one is X and Y flip pointcloud(x=-x, y=-y).
                    # Also please note that pytorch's flip function is defined on higher dimensional space,
                    # so dims=[2] means that it is flipping along the axis with H length(which is normaly the Y axis),
                    # however in our traditional word, it is flipping along the X axis.
                    # The below flip follows pytorch's definition yflip(y=-y) xflip(x=-x)
                    _, H, W, C = preds_dict[k].shape
                    preds_dict[k] = preds_dict[k].reshape(int(batch_size), 4, H, W, C)
                    preds_dict[k][:, 1] = torch.flip(preds_dict[k][:, 1], dims=[1])
                    preds_dict[k][:, 2] = torch.flip(preds_dict[k][:, 2], dims=[2])
                    preds_dict[k][:, 3] = torch.flip(preds_dict[k][:, 3], dims=[1, 2])

            if "metadata" not in example or len(example["metadata"]) == 0:
                meta_list = [None] * batch_size
            else:
                meta_list = example["metadata"]
                if double_flip:
                    meta_list = meta_list[: 4 * int(batch_size) : 4]

            batch_hm = torch.sigmoid(preds_dict["hm"])
            batch_dim = torch.exp(preds_dict["dim"])
            batch_rots = preds_dict["rot"][..., 0:1]
            batch_rotc = preds_dict["rot"][..., 1:2]
            batch_reg = preds_dict["reg"]
            batch_hei = preds_dict["height"]
            if double_flip:
                batch_hm = batch_hm.mean(dim=1)
                batch_hei = batch_hei.mean(dim=1)
                batch_dim = batch_dim.mean(dim=1)
                # y = -y reg_y = 1-reg_y
                batch_reg[:, 1, ..., 1] = 1 - batch_reg[:, 1, ..., 1]
                batch_reg[:, 2, ..., 0] = 1 - batch_reg[:, 2, ..., 0]

                batch_reg[:, 3, ..., 0] = 1 - batch_reg[:, 3, ..., 0]
                batch_reg[:, 3, ..., 1] = 1 - batch_reg[:, 3, ..., 1]
                batch_reg = batch_reg.mean(dim=1)

                # first yflip
                # y = -y theta = pi -theta
                # sin(pi-theta) = sin(theta) cos(pi-theta) = -cos(theta)
                # batch_rots[:, 1] the same
                batch_rotc[:, 1] *= -1

                # then xflip x = -x theta = 2pi - theta
                # sin(2pi - theta) = -sin(theta) cos(2pi - theta) = cos(theta)
                # batch_rots[:, 2] the same
                batch_rots[:, 2] *= -1

                # double flip
                batch_rots[:, 3] *= -1
                batch_rotc[:, 3] *= -1

                batch_rotc = batch_rotc.mean(dim=1)
                batch_rots = batch_rots.mean(dim=1)

            batch_rot = torch.atan2(batch_rots, batch_rotc)

            batch, H, W, num_cls = batch_hm.size()

            batch_reg = batch_reg.reshape(batch, H * W, 2)
            batch_hei = batch_hei.reshape(batch, H * W, 1)

            batch_rot = batch_rot.reshape(batch, H * W, 1)
            batch_dim = batch_dim.reshape(batch, H * W, 3)
            batch_hm = batch_hm.reshape(batch, H * W, num_cls)

            ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
            ys = ys.view(1, H, W).repeat(batch, 1, 1).to(batch_hm)
            xs = xs.view(1, H, W).repeat(batch, 1, 1).to(batch_hm)

            xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
            ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

            xs = xs * test_config.out_size_factor * test_config.voxel_size[0] + test_config.pc_range[0]
            ys = ys * test_config.out_size_factor * test_config.voxel_size[1] + test_config.pc_range[1]

            if "vel" in preds_dict:
                batch_vel = preds_dict["vel"]

                if double_flip:
                    # flip vy
                    batch_vel[:, 1, ..., 1] *= -1
                    # flip vx
                    batch_vel[:, 2, ..., 0] *= -1
                    batch_vel[:, 3] *= -1
                    batch_vel = batch_vel.mean(dim=1)

                batch_vel = batch_vel.reshape(batch, H * W, 2)
                batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_vel, batch_rot], dim=2)
            else:
                batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_rot], dim=2)

            metas.append(meta_list)

            if test_config.get("per_class_nms", False):
                pass
            else:
                rets.append(self.post_processing(batch_box_preds, batch_hm, test_config, post_center_range, task_id))

        # Merge branches results
        ret_list = []
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k in ["box3d_lidar", "scores"]:
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                elif k in ["label_preds"]:
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    ret[k] = torch.cat([ret[i][k] for ret in rets])

            ret["metadata"] = metas[0][i]
            ret_list.append(ret)

        results = []
        for res in ret_list:
            results.append(
                {
                    "scores": res["scores"].detach().cpu(),
                    "labels": (res["label_preds"] + 1).detach().cpu(),
                    "boxes3d": res["box3d_lidar"].detach().cpu(),
                }
            )

        return results

    @torch.no_grad()
    def post_processing(self, batch_box_preds, batch_hm, test_config, post_center_range, task_id):
        batch_size = len(batch_hm)

        prediction_dicts = []
        for i in range(batch_size):
            box_preds = batch_box_preds[i]
            hm_preds = batch_hm[i]

            scores, labels = torch.max(hm_preds, dim=-1)

            score_mask = scores > test_config.score_threshold
            distance_mask = (box_preds[..., :3] >= post_center_range[:3]).all(1) & (
                box_preds[..., :3] <= post_center_range[3:]
            ).all(1)

            mask = distance_mask & score_mask

            box_preds = box_preds[mask]
            scores = scores[mask]
            labels = labels[mask]

            boxes_for_nms = box_preds[:, [0, 1, 2, 3, 4, 5, -1]]

            if test_config.get("circular_nms", False):
                centers = boxes_for_nms[:, [0, 1]]
                boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                selected = _circle_nms(
                    boxes, min_radius=test_config.min_radius[task_id], post_max_size=test_config.nms.nms_post_max_size
                )
            else:
                selected = box_torch_ops.rotate_nms_pcdet(
                    boxes_for_nms.float(),
                    scores.float(),
                    thresh=test_config.nms.nms_iou_threshold,
                    pre_maxsize=test_config.nms.nms_pre_max_size,
                    post_max_size=test_config.nms.nms_post_max_size,
                )

            selected_boxes = box_preds[selected]
            selected_scores = scores[selected]
            selected_labels = labels[selected]

            prediction_dict = {"box3d_lidar": selected_boxes, "scores": selected_scores, "label_preds": selected_labels}

            prediction_dicts.append(prediction_dict)

        return prediction_dicts


def _circle_nms(boxes, min_radius, post_max_size=83):
    """
    NMS according to center distance
    """
    keep = np.array(circle_nms(boxes.cpu().numpy(), thresh=min_radius))[:post_max_size]
    keep = torch.from_numpy(keep).long().to(boxes.device)

    return keep
