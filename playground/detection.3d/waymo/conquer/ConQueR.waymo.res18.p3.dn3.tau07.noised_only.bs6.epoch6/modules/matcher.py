from scipy.optimize import linear_sum_assignment

import torch
from torch import nn

from .utils import box_cxcyczlwh_to_xyxyxy, generalized_box3d_iou


class HungarianMatcher3d(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, cost_rad: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_rad = cost_rad

        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0 or cost_rad != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        if "topk_indexes" in outputs.keys():
            pred_logits = torch.gather(
                outputs["pred_logits"],
                1,
                outputs["topk_indexes"].expand(-1, -1, outputs["pred_logits"].shape[-1]),
            )
            pred_boxes = torch.gather(
                outputs["pred_boxes"],
                1,
                outputs["topk_indexes"].expand(-1, -1, outputs["pred_boxes"].shape[-1]),
            )
        else:
            pred_logits = outputs["pred_logits"]
            pred_boxes = outputs["pred_boxes"]

        bs, num_queries = pred_logits.shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = pred_logits.sigmoid()
        # ([batch_size, num_queries, 6], [batch_size, num_queries, 2])
        out_bbox, out_rad = pred_boxes.split(6, dim=-1)

        # Also concat the target labels and boxes
        # [batch_size, num_target_boxes]
        tgt_ids = [v["labels"] for v in targets]
        # [batch_size, num_target_boxes, 6]
        tgt_bbox = [v["gt_boxes"][..., :6] for v in targets]
        # [batch_size, num_target_boxes, 2]
        tgt_rad = [v["gt_boxes"][..., 6:] for v in targets]

        alpha = 0.25
        gamma = 2.0

        C = []
        for i in range(bs):
            with torch.cuda.amp.autocast(enabled=False):
                out_prob_i = out_prob[i].float()
                out_bbox_i = out_bbox[i].float()
                out_rad_i = out_rad[i].float()
                tgt_bbox_i = tgt_bbox[i].float()
                tgt_rad_i = tgt_rad[i].float()

                # [num_queries, num_target_boxes]
                cost_giou = -generalized_box3d_iou(
                    box_cxcyczlwh_to_xyxyxy(out_bbox[i]),
                    box_cxcyczlwh_to_xyxyxy(tgt_bbox[i]),
                )

                neg_cost_class = (1 - alpha) * (out_prob_i**gamma) * (-(1 - out_prob_i + 1e-8).log())
                pos_cost_class = alpha * ((1 - out_prob_i) ** gamma) * (-(out_prob_i + 1e-8).log())
                cost_class = pos_cost_class[:, tgt_ids[i]] - neg_cost_class[:, tgt_ids[i]]

                # Compute the L1 cost between boxes
                # [num_queries, num_target_boxes]
                cost_bbox = torch.cdist(out_bbox_i, tgt_bbox_i, p=1)
                cost_rad = torch.cdist(out_rad_i, tgt_rad_i, p=1)

            # Final cost matrix
            C_i = (
                self.cost_bbox * cost_bbox
                + self.cost_class * cost_class
                + self.cost_giou * cost_giou
                + self.cost_rad * cost_rad
            )
            # [num_queries, num_target_boxes]
            C_i = C_i.view(num_queries, -1).cpu()
            C.append(C_i)

        indices = [linear_sum_assignment(c) for c in C]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    def extra_repr(self):
        s = "cost_class={cost_class}, cost_bbox={cost_bbox}, cost_giou={cost_giou}, cost_rad={cost_rad}"

        return s.format(**self.__dict__)
