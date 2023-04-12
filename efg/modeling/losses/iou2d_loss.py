import torch
from torch import Tensor

from efg.modeling.metrics.iou import giou, iou


def iou_loss(preds: Tensor, target: Tensor) -> Tensor:
    """Calculates the intersection over union loss.
    Args:
        preds: batch of prediction bounding boxes with representation ``[x_min, y_min, x_max, y_max]``
        target: batch of target bounding boxes with representation ``[x_min, y_min, x_max, y_max]``
    Example:
        >>> import torch
        >>> from pl_bolts.losses.object_detection import iou_loss
        >>> preds = torch.tensor([[100, 100, 200, 200]])
        >>> target = torch.tensor([[150, 150, 250, 250]])
        >>> iou_loss(preds, target)
        tensor([[0.8571]])
    Returns:
        IoU loss
    """
    loss = 1 - iou(preds, target)
    return loss


def giou_loss(preds: Tensor, target: Tensor) -> Tensor:
    """Calculates the generalized intersection over union loss.
    It has been proposed in `Generalized Intersection over Union: A Metric and A
    Loss for Bounding Box Regression <https://arxiv.org/abs/1902.09630>`_.
    Args:
        preds: an Nx4 batch of prediction bounding boxes with representation ``[x_min, y_min, x_max, y_max]``
        target: an Mx4 batch of target bounding boxes with representation ``[x_min, y_min, x_max, y_max]``
    Example:
        >>> import torch
        >>> from pl_bolts.losses.object_detection import giou_loss
        >>> preds = torch.tensor([[100, 100, 200, 200]])
        >>> target = torch.tensor([[150, 150, 250, 250]])
        >>> giou_loss(preds, target)
        tensor([[1.0794]])
    Returns:
        GIoU loss in an NxM tensor containing the pairwise GIoU loss for every element in preds and target,
        where N is the number of prediction bounding boxes and M is the number of target bounding boxes
    """
    loss = 1 - giou(preds, target)
    return loss


def iou_loss_v2(inputs, targets, weight=None, box_mode="xyxy", loss_type="iou", smooth=False, reduction="none"):
    """
    Compute iou loss of type ['iou', 'giou', 'linear_iou']

    Args:
        inputs (tensor): pred values
        targets (tensor): target values
        weight (tensor): loss weight
        box_mode (str): 'xyxy' or 'ltrb', 'ltrb' is currently supported.
        loss_type (str): 'giou' or 'iou' or 'linear_iou'
        reduction (str): reduction manner

    Returns:
        loss (tensor): computed iou loss.
    """
    if box_mode == "ltrb":
        inputs = torch.cat((-inputs[..., :2], inputs[..., 2:]), dim=-1)
        targets = torch.cat((-targets[..., :2], targets[..., 2:]), dim=-1)
    elif box_mode != "xyxy":
        raise NotImplementedError

    eps = torch.finfo(torch.float32).eps

    inputs_area = (inputs[..., 2] - inputs[..., 0]).clamp_(min=0) * (inputs[..., 3] - inputs[..., 1]).clamp_(min=0)
    targets_area = (targets[..., 2] - targets[..., 0]).clamp_(min=0) * (targets[..., 3] - targets[..., 1]).clamp_(min=0)

    w_intersect = (torch.min(inputs[..., 2], targets[..., 2]) - torch.max(inputs[..., 0], targets[..., 0])).clamp_(
        min=0
    )
    h_intersect = (torch.min(inputs[..., 3], targets[..., 3]) - torch.max(inputs[..., 1], targets[..., 1])).clamp_(
        min=0
    )

    area_intersect = w_intersect * h_intersect
    area_union = targets_area + inputs_area - area_intersect
    if smooth:
        ious = (area_intersect + 1) / (area_union + 1)
    else:
        ious = area_intersect / area_union.clamp(min=eps)

    if loss_type == "iou":
        loss = -ious.clamp(min=eps).log()
    elif loss_type == "linear_iou":
        loss = 1 - ious
    elif loss_type == "giou":
        g_w_intersect = torch.max(inputs[..., 2], targets[..., 2]) - torch.min(inputs[..., 0], targets[..., 0])
        g_h_intersect = torch.max(inputs[..., 3], targets[..., 3]) - torch.min(inputs[..., 1], targets[..., 1])
        ac_uion = g_w_intersect * g_h_intersect
        gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=eps)
        loss = 1 - gious
    else:
        raise NotImplementedError
    if weight is not None:
        loss = loss * weight.view(loss.size())
        if reduction == "mean":
            loss = loss.sum() / max(weight.sum().item(), eps)
    else:
        if reduction == "mean":
            loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()

    return loss
