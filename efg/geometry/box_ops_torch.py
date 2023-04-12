import numpy as np

import torch


def get_yaw_rotation(yaw):
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    ones = torch.ones_like(yaw)
    zeros = torch.zeros_like(yaw)

    rot = torch.stack(
        [
            torch.stack([cos_yaw, -1.0 * sin_yaw, zeros], dim=-1),
            torch.stack([sin_yaw, cos_yaw, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )

    return rot


def get_rotation_matrix(roll, pitch, yaw):
    cos_roll = torch.cos(roll)
    sin_roll = torch.sin(roll)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    cos_pitch = torch.cos(pitch)
    sin_pitch = torch.sin(pitch)

    ones = torch.ones_like(yaw)
    zeros = torch.zeros_like(yaw)

    r_roll = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cos_roll, -1.0 * sin_roll], dim=-1),
            torch.stack([zeros, sin_roll, cos_roll], dim=-1),
        ],
        dim=-2,
    )
    r_pitch = torch.stack(
        [
            torch.stack([cos_pitch, zeros, sin_pitch], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-1.0 * sin_pitch, zeros, cos_pitch], dim=-1),
        ],
        dim=-2,
    )
    r_yaw = torch.stack(
        [
            torch.stack([cos_yaw, -1.0 * sin_yaw, zeros], dim=-1),
            torch.stack([sin_yaw, cos_yaw, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )

    return torch.matmul(r_yaw, torch.matmul(r_pitch, r_roll))


def get_transform(rotation, translation):
    """Combines NxN rotation and Nx1 translation to (N+1)x(N+1) transform.
    Args:
        rotation: [..., N, N] rotation tensor.
        translation: [..., N] translation tensor. This must have the same type as
        rotation.
    Returns:
        transform: [..., (N+1), (N+1)] transform tensor. This has the same type as
        rotation.
    """
    # [..., N, 1]
    translation_n_1 = translation[..., None]
    # [..., N, N+1]
    transform = torch.cat([rotation, translation_n_1], dim=-1)
    # [..., N]
    last_row = torch.zeros_like(translation)
    # [..., N+1]
    last_row = torch.cat([last_row, torch.ones_like(last_row[..., 0:1])], dim=-1)
    # [..., N+1, N+1]
    transform = torch.cat([transform, last_row[..., None, :]], dim=-2)

    return transform


def get_upright_3d_box_corners(boxes):
    """Given a set of upright boxes, return its 8 corners.
    Given a set of boxes, returns its 8 corners. The corners are ordered layers
    (bottom, top) first and then counter-clockwise within each layer:
          5----4
         /|   /|
        6----7 |
        | 1--|-0
        |/   |/
        2----3
    Args:
        boxes: torch Tensor [N, 7]. The inner dims are [center{x,y,z}, length, width,
        height, heading].
        name: the name scope.
    Returns:
        corners: torch Tensor [N, 8, 3].
    """
    center_x, center_y, center_z, length, width, height, heading = torch.unbind(boxes, dim=-1)

    # [N, 3, 3]
    rotation = get_yaw_rotation(heading)
    # [N, 3]
    translation = torch.stack([center_x, center_y, center_z], dim=-1)

    l2 = length * 0.5
    w2 = width * 0.5
    h2 = height * 0.5

    # [N, 8, 3]
    corners = torch.reshape(
        torch.stack(
            [
                l2, w2, -h2,
                -l2, w2, -h2,
                -l2, -w2, -h2,
                l2, -w2, -h2,
                l2, w2, h2,
                -l2, w2, h2,
                -l2, -w2, h2,
                l2, -w2, h2,
            ],
            axis=-1,
        ),
        (-1, 8, 3),
    )
    # [N, 8, 3]
    corners = torch.einsum("nij,nkj->nki", rotation, corners) + translation.unsqueeze(-2)

    return corners


def is_within_box_3d(point, box):
    """Checks whether a point is in a 3d box given a set of points and boxes.
    Args:
        point: [N, 3] tensor. Inner dims are: [x, y, z].
        box: [M, 7] tensor. Inner dims are: [center_x, center_y, center_z, length,
        width, height, heading].
        name: torch name scope.
    Returns:
        point_in_box; [N, M] boolean tensor.
    """

    center = box[:, 0:3]
    dim = box[:, 3:6]
    heading = box[:, -1]
    # [M, 3, 3]
    rotation = get_yaw_rotation(heading)
    # [M, 4, 4]
    transform = get_transform(rotation, center)
    # [M, 4, 4]
    transform = torch.linalg.inv(transform)
    # [M, 3, 3]
    rotation = transform[:, 0:3, 0:3]
    # [M, 3]
    translation = transform[:, 0:3, 3]

    # [N, M, 3]
    point_in_box_frame = torch.einsum("nj,mij->nmi", point, rotation) + translation
    # [N, M, 3]
    point_in_box = torch.logical_and(
        torch.logical_and(point_in_box_frame <= dim * 0.5, point_in_box_frame >= -dim * 0.5),
        torch.all(torch.not_equal(dim, 0), -1, keepdim=True),
    )
    # [N, M]
    point_in_box = torch.prod(point_in_box.to(torch.uint8), dim=-1).to(torch.bool)

    return point_in_box


def compute_num_points_in_box_3d(point, box):
    """Computes the number of points in each box given a set of points and boxes.
    Args:
        point: [N, 3] tensor. Inner dims are: [x, y, z].
        box: [M, 7] tenosr. Inner dims are: [center_x, center_y, center_z, length,
        width, height, heading].
        name: torch name scope.
    Returns:
        num_points_in_box: [M] int32 tensor.
    """

    # [N, M]
    point_in_box = is_within_box_3d(point, box).to(torch.int32)
    num_points_in_box = torch.sum(point_in_box, dim=0)

    return num_points_in_box


def transform_point(point, from_frame_pose, to_frame_pose):
    """Transforms 3d points from one frame to another.
    Args:
        point: [..., N, 3] points.
        from_frame_pose: [..., 4, 4] origin frame poses.
        to_frame_pose: [..., 4, 4] target frame poses.
        name: torch name scope.
    Returns:
        Transformed points of shape [..., N, 3] with the same type as point.
    """
    transform = torch.linalg.matmul(torch.linalg.inv(to_frame_pose), from_frame_pose)
    return torch.einsum("...ij,...nj->...ni", transform[..., 0:3, 0:3], point) + transform[..., 0:3, 3].unsqueeze(-2)


def transform_box(box, from_frame_pose, to_frame_pose):
    """Transforms 3d upright boxes from one frame to another.
    Args:
        box: [..., N, 7] boxes.
        from_frame_pose: [...,4, 4] origin frame poses.
        to_frame_pose: [...,4, 4] target frame poses.
        name: torch name scope.
    Returns:
        Transformed boxes of shape [..., N, 7] with the same type as box.
    """
    transform = torch.linalg.matmul(torch.linalg.inv(to_frame_pose), from_frame_pose)
    heading_offset = torch.atan2(transform[..., 1, 0], transform[..., 0, 0])
    heading = box[..., -1] + heading_offset[..., None]
    center = torch.einsum("...ij,...nj->...ni", transform[..., 0:3, 0:3], box[..., 0:3]) + transform[
        ..., 0:3, 3
    ].unsqueeze(-2)

    return torch.cat([center, box[..., 3:6], heading[..., None]], dim=-1)


def limit_period(val, offset=0.5, period=np.pi):
    """Limit the value into a period for periodic function.
    Args:
        val (np.ndarray): The value to be converted.
        offset (float, optional): Offset to set the value range. \
            Defaults to 0.5.
        period (float, optional): Period of the value. Defaults to np.pi.
    Returns:
        torch.Tensor: Value in the range of \
            [-offset * period, (1-offset) * period)
    """
    if isinstance(val, torch.Tensor):
        is_tensor = True
    elif isinstance(val, np.ndarray):
        is_tensor = False
        val = torch.from_numpy(val).float()
    else:
        raise ValueError("Only support tensor or ndarray!")

    val = val - torch.floor(val / period + offset) * period

    if not ((val >= -offset * period) & (val <= offset * period)).all().item():
        val = torch.clamp(val, min=-offset * period, max=offset * period)

    return val if is_tensor else val.numpy()
