import numpy as np

import open3d as o3d

# order in which bbox vertices will be connected
LINE_SEGMENTS = [
    [0, 1], [1, 3], [3, 2], [2, 0],
    [4, 5], [5, 7], [7, 6], [6, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
]


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
        axis=-2,
    )
    return rot


def get_bbox(label) -> np.ndarray:
    # label: x, y, z, l, w, h, yaw
    width = label[4]
    length = label[3]

    return np.array(
        [
            [-0.5 * length, -0.5 * width],
            [-0.5 * length, 0.5 * width],
            [0.5 * length, -0.5 * width],
            [0.5 * length, 0.5 * width],
        ]
    )


def transform_bbox_waymo(label) -> np.ndarray:
    """Transform object's 3D bounding box using Waymo utils"""
    heading = -label[-1]
    bbox_corners = get_bbox(label)

    mat = get_yaw_rotation(heading)
    rot_mat = mat[:2, :2]

    return bbox_corners @ rot_mat


def transform_bbox_custom(label) -> np.ndarray:
    """Transform object's 3D bounding box without Waymo utils"""
    heading = -label.box.heading
    bbox_corners = get_bbox(label)
    rot_mat = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])

    return bbox_corners @ rot_mat


def build_open3d_bbox(box: np.ndarray, label: np.ndarray):
    """Create bounding box's points and lines needed for drawing in open3d"""
    x = label[0]
    y = label[1]
    z = label[2]

    z_bottom = z - label[5] / 2
    z_top = z + label[5] / 2

    points = [[0.0, 0.0, 0.0]] * box.shape[0] * 2
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
    opt.point_size = 5.0

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


if __name__ == "__main__":
    # waymo visualization
    points = np.load("example_data/example_point_cloud.bin.npy")[:, :3]
    boxes = np.load("example_data/example_point_cloud_boxes.bin.npy")[:, [0, 1, 2, 3, 4, 5, -1]]
    gt_boxes = np.load("example_data/example_point_cloud_boxes_gt.bin.npy")

    show_point_cloud(points, gt_boxes, boxes)
