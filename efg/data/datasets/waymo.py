import collections
import logging
import os
import pickle
from copy import deepcopy

import numpy as np

import torch

from efg.data.augmentations import build_processors
from efg.data.augmentations3d import _dict_select
from efg.data.base_dataset import BaseDataset
from efg.data.datasets.utils import read_single_waymo, read_single_waymo_sweep
from efg.data.registry import DATASETS
from efg.utils.file_io import PathManager

logger = logging.getLogger(__name__)

CAT_TO_IDX = {
    "UNKNOWN": 0,
    "VEHICLE": 1,
    "PEDESTRIAN": 2,
    "SIGN": 3,
    "CYCLIST": 4,
}
IDX_TO_CAT = ["UNKNOWN", "VEHICLE", "PEDESTRIAN", "SIGN", "CYCLIST"]
# ignore sign class
LABEL_TO_TYPE = {1: 1, 2: 2, 3: 4}


@DATASETS.register()
class WaymoDetectionDataset(BaseDataset):
    def __init__(self, config):
        super(WaymoDetectionDataset, self).__init__(config)

        self.is_test = config.task == "test"
        self.class_names = config.dataset.classes
        self.load_interval = config.dataset.load_interval
        self.nsweeps = config.dataset.nsweeps
        self.num_point_features = len(config.dataset.format) if self.nsweeps == 1 else len(config.dataset.format) + 1
        logger.info(f"Using {self.nsweeps} sweep(s)")

        dataset_config = config.dataset.source
        root_path = dataset_config.root
        info_file = dataset_config[config.task]

        self.root_path = root_path
        self.info_path = root_path + info_file
        self.db_path = self.info_path.split("/infos")[0]

        self.dataset_dicts = self.load_infos()
        logger.info(f"Using {len(self.dataset_dicts)} frames for {self.config.task}")

        self.transforms = build_processors(config.dataset.processors[config.task])
        logger.info(f"Building data processors: {self.transforms}")

    def load_infos(self):
        waymo_infos_all = pickle.load(PathManager.open(self.info_path, "rb"))
        return waymo_infos_all[:: self.load_interval]

    def __len__(self):
        return len(self.dataset_dicts)

    def __getitem__(self, idx):
        info = deepcopy(self.dataset_dicts[idx])

        # load point cloud data
        if not os.path.isabs(info["path"]):
            info["path"] = os.path.join(self.root_path, info["path"])
        obj = pickle.load(PathManager.open(info["path"], "rb"))
        points = read_single_waymo(obj)

        nsweeps = self.nsweeps
        if nsweeps > 1:
            sweep_points_list = [points]
            sweep_times_list = [np.zeros((points.shape[0], 1))]

            assert (nsweeps - 1) <= len(info["sweeps"]), "nsweeps {} should be equal to the list length {}.".format(
                nsweeps, len(info["sweeps"])
            )

            for i in range(nsweeps - 1):
                sweep = info["sweeps"][i]
                sweep_obj = pickle.load(PathManager.open(sweep["path"], "rb"))
                points_sweep, times_sweep = read_single_waymo_sweep(sweep, sweep_obj)
                sweep_points_list.append(points_sweep)
                sweep_times_list.append(times_sweep)

            points = np.concatenate(sweep_points_list, axis=0)
            times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

            points = np.hstack([points, times])

        info["metadata"] = {
            "root_path": self.root_path,
            "db_path": self.db_path,
            "token": info["token"],
            "num_point_features": self.num_point_features,
        }

        if not self.is_test:
            if "annotations" not in info:
                info["annotations"] = {
                    "gt_boxes": info.pop("gt_boxes").astype(np.float32),
                    "gt_names": info.pop("gt_names"),
                    "difficulty": info.pop("difficulty").astype(np.int8),
                    "num_points_in_gt": info.pop("num_points_in_gt").astype(np.int64),
                }

            self._filter_gt_by_classes(info)
            if len(info["sweeps"]) > 0 and "annotations" in info["sweeps"][0]:
                [self._filter_gt_by_classes(s) for s in info["sweeps"]]

        points, info = self._apply_transforms(points, info)

        if not self.is_test:
            self._add_class_labels_to_annos(info)
            if len(info["sweeps"]) > 0 and "annotations" in info["sweeps"][0]:
                [self._add_class_labels_to_annos(s) for s in info["sweeps"]]

        return points, info

    def _filter_gt_by_classes(self, info):
        target = info["annotations"]
        keep = (target["gt_names"][:, None] == self.class_names).any(axis=1)
        _dict_select(target, keep)

    def _add_class_labels_to_annos(self, info):
        info["annotations"]["labels"] = (
            np.array([self.class_names.index(name) + 1 for name in info["annotations"]["gt_names"]])
            .astype(np.int64)
            .reshape(-1)
        )


def collate(batch_list, device):
    targets_merged = collections.defaultdict(list)
    for targets in batch_list:
        for k, v in targets.items():
            targets_merged[k].append(v)
    batch_size = len(batch_list)

    ret = {}
    for key, elems in targets_merged.items():
        if key in ["voxels", "num_points_per_voxel", "num_voxels"]:
            ret[key] = torch.tensor(np.concatenate(elems, axis=0)).to(device)
        elif key in ["gt_boxes", "labels", "gt_names", "difficulty", "num_points_in_gt"]:
            max_gt = -1
            for k in range(batch_size):
                max_gt = max(max_gt, len(elems[k]))
                batch_gt_boxes3d = np.zeros((batch_size, max_gt, *elems[0].shape[1:]), dtype=elems[0].dtype)
            for i in range(batch_size):
                batch_gt_boxes3d[i, : len(elems[i])] = elems[i]
            if key != "gt_names":
                batch_gt_boxes3d = torch.tensor(batch_gt_boxes3d, device=device)
            ret[key] = batch_gt_boxes3d
        elif key == "calib":
            ret[key] = {}
            for elem in elems:
                for k1, v1 in elem.items():
                    if k1 not in ret[key]:
                        ret[key][k1] = [v1]
                    else:
                        ret[key][k1].append(v1)
            for k1, v1 in ret[key].items():
                ret[key][k1] = torch.tensor(np.stack(v1, axis=0))
        elif key in ["coordinates", "points"]:
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode="constant", constant_values=i)
                coors.append(coor_pad)
            ret[key] = torch.tensor(np.concatenate(coors, axis=0)).to(device)
        else:
            ret[key] = np.stack(elems, axis=0)

    return ret
