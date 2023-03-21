import numpy as np

import open3d as o3d

# from pyquaternion import Quaternion
# from nuscenes.utils.data_classes import Box
# from nuscenes.eval.common.utils import quaternion_yaw

# order in which bbox vertices will be connected
LINE_SEGMENTS = [[0, 1], [1, 3], [3, 2], [2, 0],
                 [4, 5], [5, 7], [7, 6], [6, 4],
                 [0, 4], [1, 5], [2, 6], [3, 7]]


def get_yaw_rotation(yaw):
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    ones = np.ones_like(yaw)
    zeros = np.zeros_like(yaw)
    rot = np.stack(
        [
            np.stack([cos_yaw, -1.0 * sin_yaw, zeros], axis=-1),
            np.stack([sin_yaw, cos_yaw, zeros], axis=-1),
            np.stack([zeros, zeros, ones], axis=-1),
        ],
        axis=-2
    )
    return rot


def get_bbox(label) -> np.ndarray:
    # label: x, y, z, l, w, h, yaw
    width = label[4]
    length = label[3]

    return np.array([
        [-0.5 * length, -0.5 * width],
        [-0.5 * length, 0.5 * width],
        [0.5 * length, -0.5 * width],
        [0.5 * length, 0.5 * width],
    ])


def transform_bbox_waymo(label) -> np.ndarray:
    """Transform object's 3D bounding box using Waymo utils"""
    heading = - label[-1]
    bbox_corners = get_bbox(label)

    mat = get_yaw_rotation(heading)
    rot_mat = mat[:2, :2]

    return bbox_corners @ rot_mat


def transform_bbox_custom(label) -> np.ndarray:
    """Transform object's 3D bounding box without Waymo utils"""
    heading = -label.box.heading
    bbox_corners = get_bbox(label)
    rot_mat = np.array([[np.cos(heading), - np.sin(heading)],
                        [np.sin(heading), np.cos(heading)]])

    return bbox_corners @ rot_mat


def build_open3d_bbox(box: np.ndarray, label: np.ndarray):
    """Create bounding box's points and lines needed for drawing in open3d"""
    x = label[0]
    y = label[1]
    z = label[2]

    z_bottom = z - label[5] / 2
    z_top = z + label[5] / 2

    points = [[0., 0., 0.]] * box.shape[0] * 2
    for idx in range(box.shape[0]):
        x_, y_ = x + box[idx][0], y + box[idx][1]
        points[idx] = [x_, y_, z_bottom]
        points[idx + 4] = [x_, y_, z_top]

    return points


def show_point_cloud(points: np.ndarray, laser_labels: np.ndarray, gt_labels: np.ndarray) -> None:
    # pylint: disable=no-member (E1101)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])  # set background color to white
    opt.point_size = 5.

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0, 0, 0])  # set points color to black
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.6, origin=[0, 0, 0])

    vis.add_geometry(pcd)
    vis.add_geometry(mesh_frame)

    for label in laser_labels:
        bbox_corners = transform_bbox_waymo(label)
        # bbox_corners = transform_bbox_custom(label)
        bbox_points = build_open3d_bbox(bbox_corners, label)

        colors = [[0, 0, 1] for _ in range(len(LINE_SEGMENTS))]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(bbox_points),
            lines=o3d.utility.Vector2iVector(LINE_SEGMENTS),
        )

        line_set.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(line_set)

    for label in gt_labels:
        bbox_corners = transform_bbox_waymo(label)
        # bbox_corners = transform_bbox_custom(label)
        bbox_points = build_open3d_bbox(bbox_corners, label)

        colors = [[0, 1, 0] for _ in range(len(LINE_SEGMENTS))]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(bbox_points),
            lines=o3d.utility.Vector2iVector(LINE_SEGMENTS),
        )

        line_set.colors = o3d.utility.Vector3dVector(colors)

        vis.add_geometry(line_set)

    vis.run()


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
    transform = np.matmul(np.linalg.inv(to_frame_pose), from_frame_pose)
    return np.einsum('...ij,...nj->...ni', transform[..., 0:3, 0:3], point) + \
        np.expand_dims(transform[..., 0:3, 3], axis=-2)


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
    transform = np.matmul(np.linalg.inv(to_frame_pose), from_frame_pose)
    heading_offset = np.atan2(transform[..., 1, 0], transform[..., 0, 0])
    heading = box[..., -1] + np.expand_dims(heading_offset, axis=-1)
    center = np.einsum('...ij,...nj->...ni', transform[..., 0:3, 0:3], box[..., 0:3]) \
        + transform[..., 0:3, 3].expand_dims(-2)

    return np.concatenate([center, box[..., 3:6], np.expand_dims(heading, axis=-1)], axis=-1)


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
    center_x, center_y, center_z, length, width, height, heading = np.split(boxes, axis=-1)

    # [N, 3, 3]
    rotation = get_yaw_rotation(heading)
    # [N, 3]
    translation = np.stack([center_x, center_y, center_z], axis=-1)

    l2 = length * 0.5
    w2 = width * 0.5
    h2 = height * 0.5

    # [N, 8, 3]
    corners = np.reshape(
        np.stack([
            l2, w2, -h2,
            -l2, w2, -h2,
            -l2, -w2, -h2,
            l2, -w2, -h2,
            l2, w2, h2,
            -l2, w2, h2,
            -l2, -w2, h2,
            l2, -w2, h2
        ], axis=-1),
        (-1, 8, 3)
    )
    # [N, 8, 3]
    corners = np.einsum('nij,nkj->nki', rotation, corners) + np.expand_dims(translation, axis=-2)

    return corners


if __name__ == "__main__":

    # waymo visualization
    points = np.load("example_data/example_point_cloud.bin.npy")[:, :3]
    boxes = np.load("example_data/example_point_cloud_boxes.bin.npy")[:, [0, 1, 2, 3, 4, 5, -1]]
    gt_boxes = np.load("example_data/example_point_cloud_boxes_gt.bin.npy")

    show_point_cloud(points, gt_boxes, boxes)
