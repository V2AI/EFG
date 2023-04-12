import torch
from torch import nn
from torch.nn import functional as F

from efg.modeling.losses.focal_loss import sigmoid_focal_loss_jit as sigmoid_focal_loss
from efg.utils.distributed import get_world_size

from modules.utils import box_cxcyczlwh_to_xyxyxy, generalized_box3d_iou


def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def _get_tgt_permutation_idx(indices):
    # permute targets following indices
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx


class ClassificationLoss(nn.Module):
    def __init__(self, focal_alpha):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.target_classes = None
        self.src_logits = None

    def forward(self, outputs, targets, indices, num_boxes):
        # put indices into outputs
        outputs["matched_indices"] = indices

        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        target_classes_onehot = torch.zeros_like(src_logits)

        idx = _get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])  # 0, 1, 2

        # for metrics calculation
        self.target_classes = target_classes_o

        if "topk_indexes" in outputs.keys():
            topk_indexes = outputs["topk_indexes"]
            self.src_logits = torch.gather(src_logits, 1, topk_indexes.expand(-1, -1, src_logits.shape[-1]))[idx]
            target_classes_onehot[idx[0], topk_indexes[idx].squeeze(-1), target_classes_o] = 1
        else:
            self.src_logits = src_logits[idx]
            # 0 for bg, 1 for fg
            # N, L, C
            target_classes_onehot[idx[0], idx[1], target_classes_o] = 1

        loss_ce = (
            sigmoid_focal_loss(
                src_logits,
                target_classes_onehot,
                alpha=self.focal_alpha,
                gamma=2.0,
                reduction="sum",
            )
            / num_boxes
        )

        losses = {
            "loss_ce": loss_ce,
        }

        return losses


class RegressionLoss(nn.Module):
    def forward(self, outputs, targets, indices, num_boxes):
        assert "pred_boxes" in outputs
        idx = _get_src_permutation_idx(indices)

        if "topk_indexes" in outputs.keys():
            pred_boxes = torch.gather(
                outputs["pred_boxes"],
                1,
                outputs["topk_indexes"].expand(-1, -1, outputs["pred_boxes"].shape[-1]),
            )
        else:
            pred_boxes = outputs["pred_boxes"]
        target_boxes = torch.cat([t["gt_boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        src_boxes, src_rads = pred_boxes[idx].split(6, dim=-1)
        target_boxes, target_rads = target_boxes.split(6, dim=-1)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_rad = F.l1_loss(src_rads, target_rads, reduction="none")

        loss_giou = 1 - torch.diag(
            generalized_box3d_iou(
                box_cxcyczlwh_to_xyxyxy(src_boxes),
                box_cxcyczlwh_to_xyxyxy(target_boxes),
            )
        )

        losses = {
            "loss_bbox": loss_bbox.sum() / num_boxes,
            "loss_giou": loss_giou.sum() / num_boxes,
            "loss_rad": loss_rad.sum() / num_boxes,
        }

        return losses


class Det3DLoss(nn.Module):
    def __init__(self, matcher, weight_dict, losses):
        super().__init__()

        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

        self.det3d_losses = nn.ModuleDict()
        self.det3d_enc_losses = nn.ModuleDict()
        for loss in losses:
            if loss == "boxes":
                self.det3d_losses[loss] = RegressionLoss()
                self.det3d_enc_losses[loss + "_enc"] = RegressionLoss()
            elif loss == "focal_labels":
                self.det3d_losses[loss] = ClassificationLoss(0.25)
                self.det3d_enc_losses[loss + "_enc"] = ClassificationLoss(0.25)
            else:
                raise ValueError(f"Only boxes|focal_labels are supported for det3d losses. Found {loss}")

    def get_target_classes(self):
        for k in self.det3d_losses.keys():
            if "labels" in k:
                return self.det3d_losses[k].src_logits, self.det3d_losses[k].target_classes

    def prep_for_dn(self, dn_meta):
        output_known_lbs_bboxes = dn_meta["output_known_lbs_bboxes"]
        num_dn_groups, pad_size = dn_meta["num_dn_group"], dn_meta["pad_size"]
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size // num_dn_groups

        return output_known_lbs_bboxes, single_pad, num_dn_groups

    def forward(self, outputs, targets, dn_meta=None):
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum([len(t["labels"]) for t in targets])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if get_world_size() > 1:
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}

        if dn_meta is not None:
            # prepare for computing denosing loss
            output_known_lbs_bboxes, single_pad, scalar = self.prep_for_dn(dn_meta)
            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]["labels"]) > 0:
                    t = torch.arange(0, len(targets[i]["labels"]) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            l_dict = {}
            for loss in self.losses:
                l_dict.update(
                    self.det3d_losses[loss](
                        output_known_lbs_bboxes,
                        targets,
                        dn_pos_idx,
                        num_boxes * scalar,
                    )
                )
            l_dict = {k + "_dn": v for k, v in l_dict.items()}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.det3d_losses[loss](aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if dn_meta is not None:
                    aux_outputs_known = output_known_lbs_bboxes["aux_outputs"][i]
                    l_dict = {}
                    for loss in self.losses:
                        l_dict.update(
                            self.det3d_losses[loss](
                                aux_outputs_known,
                                targets,
                                dn_pos_idx,
                                num_boxes * scalar,
                            )
                        )
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)
        for loss in self.losses:
            losses.update(self.det3d_losses[loss](outputs, targets, indices, num_boxes))

        return losses
