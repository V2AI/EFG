import torch
import numpy as np
from torch import nn
from modules.utils import rotate_points_along_z, boxes_to_corners_3d


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """

    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta

        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()
        else:
            self.code_weights = None

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n**2 / beta, n - 0.5 * beta)

        return loss

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None
    ):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert (
                weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            )
            loss = loss * weights.unsqueeze(-1)

        return loss


def get_corner_loss_lidar(pred_bbox3d: torch.Tensor, gt_bbox3d: torch.Tensor):
    """
    Args:
        pred_bbox3d: (N, 7) float Tensor.
        gt_bbox3d: (N, 7) float Tensor.

    Returns:
        corner_loss: (N) float Tensor.
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

    pred_box_corners = boxes_to_corners_3d(pred_bbox3d)
    gt_box_corners = boxes_to_corners_3d(gt_bbox3d)
    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi
    gt_box_corners_flip = boxes_to_corners_3d(gt_bbox3d_flip)
    # (N, 8)
    corner_dist = torch.min(
        torch.norm(pred_box_corners - gt_box_corners, dim=2),
        torch.norm(pred_box_corners - gt_box_corners_flip, dim=2),
    )
    # (N, 8)
    corner_loss = WeightedSmoothL1Loss.smooth_l1_loss(corner_dist, beta=1.0)

    return corner_loss.mean(dim=1)


def get_corner_loss(rcnn_reg, roi_boxes3d, gt_of_rois_src, fg_mask):
    fg_rcnn_reg = rcnn_reg[fg_mask]
    fg_roi_boxes3d = roi_boxes3d[fg_mask]
    code_size = 7
    fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
    batch_anchors = fg_roi_boxes3d.clone().detach()
    roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
    roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
    batch_anchors[:, :, 0:3] = 0
    rcnn_boxes3d = decode_torch(
        fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
    ).view(-1, code_size)

    rcnn_boxes3d = rotate_points_along_z(rcnn_boxes3d.unsqueeze(dim=1), roi_ry).squeeze(
        dim=1
    )
    rcnn_boxes3d[:, 0:3] += roi_xyz

    loss_corner = get_corner_loss_lidar(
        rcnn_boxes3d[:, 0:7], gt_of_rois_src[fg_mask][:, 0:7]
    )
    loss_corner = loss_corner.mean()

    return loss_corner


def decode_torch(box_encodings, anchors):
    """
    Args:
        box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
        anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

    Returns:

    """

    encode_angle_by_sincos = False
    xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
    if not encode_angle_by_sincos:
        xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(box_encodings, 1, dim=-1)
    else:
        xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(
            box_encodings, 1, dim=-1
        )

    diagonal = torch.sqrt(dxa**2 + dya**2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * dza + za

    dxg = torch.exp(dxt) * dxa
    dyg = torch.exp(dyt) * dya
    dzg = torch.exp(dzt) * dza

    if encode_angle_by_sincos:
        rg_cos = cost + torch.cos(ra)
        rg_sin = sint + torch.sin(ra)
        rg = torch.atan2(rg_sin, rg_cos)
    else:
        rg = rt + ra

    cgs = [t + a for t, a in zip(cts, cas)]
    return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)
