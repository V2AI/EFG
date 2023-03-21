import os
import pickle

import numpy as np


def read_from_file(info, nsweeps=1, root_path=""):

    if not os.path.isabs(info["path"]):
        info["path"] = os.path.join(root_path, info["path"])
    with open(info["path"], "rb") as f:
        obj = pickle.load(f)

    points = read_single_waymo(obj)
    times = None

    if nsweeps > 1:
        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        assert (nsweeps - 1) == len(
            info["sweeps"]
        ), "nsweeps {} should be equal to the list length {}.".format(
            nsweeps, len(info["sweeps"])
        )

        for i in range(nsweeps - 1):
            sweep = info["sweeps"][i]
            if not os.path.isabs(sweep["path"]):
                sweep["path"] = os.path.join(root_path, sweep["path"])
            with open(sweep["path"], "rb") as f:
                sweep_obj = pickle.load(f)
            points_sweep, times_sweep = read_single_waymo_sweep(sweep, sweep_obj)

            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

    if times is not None:
        points = np.hstack([points, times])

    return points


def read_pc_annotations(info):
    if "annotations" in info:
        annos_dict = info["annotations"]
        annos = {
            "boxes": annos_dict["gt_boxes"].astype(np.float32),
            "names": annos_dict["gt_names"],
            "difficulty": annos_dict["difficulty"].astype(np.int8),
            "num_points_in_gt": annos_dict["num_points_in_gt"].astype(np.int64),
        }
        return annos
    elif "gt_boxes" in info:
        annos = {
            "boxes": info["gt_boxes"].astype(np.float32),
            "names": info["gt_names"],
            "difficulty": info["difficulty"].astype(np.int8),
            "num_points_in_gt": info["num_points_in_gt"].astype(np.int64),
        }
        return annos
    else:
        return None


def read_single_waymo(obj):
    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity
    points_feature[:, 0] = np.tanh(points_feature[:, 0])

    points = np.concatenate([points_xyz, points_feature], axis=-1)

    return points


def read_single_waymo_sweep(sweep, obj):

    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity
    points_feature[:, 0] = np.tanh(points_feature[:, 0])
    points_sweep = np.concatenate([points_xyz, points_feature], axis=-1).T  # 5 x N

    num_points = points_sweep.shape[1]

    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(
            np.vstack((points_sweep[:3, :], np.ones(num_points)))
        )[:3, :]

    cur_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, cur_times.T


def get_start_result_anno():
    annotations = {}
    annotations.update({
        # 'index': None,
        "name": [],
        "truncated": [],
        "occluded": [],
        "alpha": [],
        "bbox": [],
        "dimensions": [],
        "location": [],
        "rotation_y": [],
        "score": [],
    })
    return annotations


def empty_result_anno():
    annotations = {}
    annotations.update({
        "name": np.array([]),
        "truncated": np.array([]),
        "occluded": np.array([]),
        "alpha": np.array([]),
        "bbox": np.zeros([0, 4]),
        "dimensions": np.zeros([0, 3]),
        "location": np.zeros([0, 3]),
        "rotation_y": np.array([]),
        "score": np.array([]),
    })
    return annotations
