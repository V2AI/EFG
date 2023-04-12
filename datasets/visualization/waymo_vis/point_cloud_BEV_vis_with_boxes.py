import numpy as np
from pyquaternion import Quaternion

from nuscenes.utils.data_classes import Box

from matplotlib import pyplot as plt

# plt.rcParams['image.cmap'] = 'twilight'
plt.rcParams["image.cmap"] = "Greys"


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[: view.shape[0], : view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


if __name__ == "__main__":
    points = np.load("example_data/example_point_cloud.bin.npy")[:, :3].transpose()
    boxes = np.load("example_data/example_point_cloud_boxes.bin.npy")
    gt_boxes = np.load("example_data/example_point_cloud_boxes_gt.bin.npy")

    eval_range = 54.0

    # Init axes.
    _, ax = plt.subplots(1, 1, figsize=(11, 11))

    # Show point cloud.
    points = view_points(points[:3, :], np.eye(4), normalize=False)
    dists = np.sqrt(np.sum(points[:2, :] ** 2, axis=0))
    colors = np.minimum(1, dists / eval_range)
    colors = np.zeros_like(colors)
    ax.scatter(points[0, :], points[1, :], c="0.7", s=0.1)

    # Show ego vehicle.
    ax.plot(0, 0, "x", color="black")

    # Show GT boxes.
    for raw_box in gt_boxes:
        box = Box(
            raw_box[:3],
            raw_box[[4, 3, 5]],
            Quaternion(axis=[0, 0, 1], radians=raw_box[-1]),
            label=1,
            score=1,
            velocity=(*raw_box[6:8], 0),
        )
        box.render(ax, view=np.eye(4), colors=("g", "g", "g"), linewidth=0.6)

    # Show EST boxes.
    for raw_box in boxes:
        box = Box(
            raw_box[:3],
            raw_box[[4, 3, 5]],
            Quaternion(axis=[0, 0, 1], radians=raw_box[-1]),
            label=1,
            score=1,
            velocity=(*raw_box[6:8], 0),
        )
        box.render(ax, view=np.eye(4), colors=("b", "b", "b"), linewidth=0.3)

    # Limit visible range.
    axes_limit = eval_range + 3  # Slightly bigger to include boxes that extend beyond the range.
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)

    plt.title("sample scene")
    # plt.savefig('filename.png', dpi=1024)
    plt.show()
