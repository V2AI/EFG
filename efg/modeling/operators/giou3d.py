import torch

from efg.geometry.box_ops_torch import get_upright_3d_box_corners
from efg.modeling.operators.iou_box3d import box3d_overlap


def generalized_iou_box3d(boxes1, boxes2):
    """
    from RHC      Y
        (5) +-----|---+. (4)
            | ` . |   |  ` .
            | (6) +---+-----+ (7)
      Z     |     |   |     |
        (1) +-----+---+. (0)|
            ` .   |     ` . |
            (2) ` +---------+ (3) ----- X
    to
       Y
        (4) +---------+. (5)
            | ` .     |  ` .
            | (0) +---+-----+ (1)------ X
            |     |   |     |
        (7) +-----+---+. (6)|
            ` .   |     ` . |
            (3) ` +---------+ (2)
                  |
                  Z
    """
    boxes1 = boxes1.view(-1, boxes1.shape[-1])
    boxes1_corners = get_upright_3d_box_corners(boxes1)
    boxes1_corners = boxes1_corners[:, [2, 3, 0, 1, 6, 7, 4, 5], :]
    boxes2 = boxes2.view(-1, boxes2.shape[-1])
    boxes2_corners = get_upright_3d_box_corners(boxes2)
    boxes2_corners = boxes2_corners[:, [2, 3, 0, 1, 6, 7, 4, 5], :]

    # intersection volumes; ious
    inter_vols, inter_ious = box3d_overlap(boxes1_corners, boxes2_corners)
    enclosing_vols = enclosing_box3d_vol(boxes1_corners, boxes2_corners)
    # vols of boxes
    vols1 = box3d_vol_tensor(boxes1_corners)
    vols2 = box3d_vol_tensor(boxes2_corners)
    sum_vols = vols1[:, None] + vols2[None, :]
    # gIOU = iou - (1 - sum_vols/enclose_vol)
    union_vols = sum_vols - inter_vols
    gious = inter_ious - (1 - union_vols / enclosing_vols)

    return inter_ious, gious


def box3d_vol_tensor(corners):
    EPS = 1e-6
    reshape = False
    B, K = corners.shape[0], corners.shape[1]
    if len(corners.shape) == 4:
        # batch x prop x 8 x 3
        reshape = True
        corners = corners.view(-1, 8, 3)
    a = torch.sqrt((corners[:, 0, :] - corners[:, 1, :]).pow(2).sum(dim=1).clamp(min=EPS))
    b = torch.sqrt((corners[:, 1, :] - corners[:, 2, :]).pow(2).sum(dim=1).clamp(min=EPS))
    c = torch.sqrt((corners[:, 0, :] - corners[:, 4, :]).pow(2).sum(dim=1).clamp(min=EPS))
    vols = a * b * c
    if reshape:
        vols = vols.view(B, K)

    return vols


def enclosing_box3d_vol(corners1, corners2):
    """
    volume of enclosing axis-aligned box
    """
    # EPS = 1e-6

    al_xmin = torch.min(
        torch.min(corners1[..., 0], dim=1).values[:, None],
        torch.min(corners2[..., 0], dim=1).values[None, :],
    )
    al_ymin = torch.max(
        torch.max(corners1[..., 1], dim=1).values[:, None],
        torch.max(corners2[..., 1], dim=1).values[None, :],
    )
    al_zmin = torch.min(
        torch.min(corners1[..., 2], dim=1).values[:, None],
        torch.min(corners2[..., 2], dim=1).values[None, :],
    )
    al_xmax = torch.max(
        torch.max(corners1[..., 0], dim=1).values[:, None],
        torch.max(corners2[..., 0], dim=1).values[None, :],
    )
    al_ymax = torch.min(
        torch.min(corners1[..., 1], dim=1).values[:, None],
        torch.min(corners2[..., 1], dim=1).values[None, :],
    )
    al_zmax = torch.max(
        torch.max(corners1[..., 2], dim=1).values[:, None],
        torch.max(corners2[..., 2], dim=1).values[None, :],
    )

    diff_x = torch.abs(al_xmax - al_xmin)
    diff_y = torch.abs(al_ymax - al_ymin)
    diff_z = torch.abs(al_zmax - al_zmin)
    vol = diff_x * diff_y * diff_z

    return vol
