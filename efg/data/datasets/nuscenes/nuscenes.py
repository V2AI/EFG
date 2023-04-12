import logging
import os
import pickle
from copy import deepcopy

import numpy as np

from efg.data.augmentations import build_processors
from efg.data.augmentations3d import _dict_select
from efg.data.base_dataset import BaseDataset
from efg.data.datasets.nuscenes.nusc_common import cls_attr_dist, general_to_detection, read_file, read_sweep
from efg.data.registry import DATASETS
from efg.utils.file_io import PathManager

logger = logging.getLogger(__name__)

fit_plane_LSE_RANSAC = None


def drop_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


@DATASETS.register()
class nuScenesDetectionDataset(BaseDataset):
    def __init__(self, config):
        super(nuScenesDetectionDataset, self).__init__(config)
        self.config = config

        self.load_interval = config.dataset.load_interval

        self.nsweeps = config.dataset.nsweeps
        logger.info(f"Using {self.nsweeps} sweep(s)")

        if config.task == "train":
            dataset_config = config.dataset.source
        elif config.task == "val":
            dataset_config = config.dataset.eval_source

        root_path = dataset_config.root
        info_file = dataset_config[config.task]

        self.is_train = config.task == "train"

        self.root_path = root_path
        self.info_path = self.root_path + info_file
        self.db_path = self.info_path.split("/infos")[0]

        self.class_names = config.dataset.classes
        self._name_mapping = general_to_detection

        self.num_point_features = len(config.dataset.format)

        self.meta = self._get_metadata()

        self.dataset_dicts = self.load_infos()
        logger.info(f"Using {len(self.dataset_dicts)} frames for {self.config.task}")

        self.transforms = build_processors(config.dataset.processors[config.task])
        logger.info(f"Building data processors: {self.transforms}")

        self.version = config.dataset.nusc.version
        self.eval_version = config.dataset.nusc.eval_version

    def _get_metadata(self):
        mapped_class_names = []
        for n in self.config.dataset.classes:
            if n in general_to_detection:
                mapped_class_names.append(general_to_detection[n])
            else:
                mapped_class_names.append(n)
        meta = {
            "class_names": self.config.dataset.classes,
            "mapped_class_names": mapped_class_names,
            "general_to_detection": general_to_detection,
            "cls_attr_dist": cls_attr_dist,
            "nsweeps": self.config.dataset.nsweeps,
            "evaluator_type": self.config.dataset.nusc.eval_version,
            "version": self.config.dataset.nusc.version,
            "root_path": self.root_path,
        }
        return meta

    def load_infos(self):
        _nusc_infos_all = pickle.load(PathManager.open(self.info_path, "rb"))

        if self.is_train:  # if training
            self.frac = int(len(_nusc_infos_all) * 0.25)

            _cls_infos = {name: [] for name in self.meta["class_names"]}
            for info in _nusc_infos_all:
                for name in set(info["gt_names"]):
                    if name in self.meta["class_names"]:
                        _cls_infos[name].append(info)
            duplicated_samples = sum([len(v) for _, v in _cls_infos.items()])
            _cls_dist = {k: len(v) / duplicated_samples for k, v in _cls_infos.items()}

            _nusc_infos = []
            frac = 1.0 / len(self.meta["class_names"])
            ratios = [frac / v for v in _cls_dist.values()]
            for cls_infos, ratio in zip(list(_cls_infos.values()), ratios):
                _nusc_infos += np.random.choice(cls_infos, int(len(cls_infos) * ratio)).tolist()
            _cls_infos = {name: [] for name in self.meta["class_names"]}
            for info in _nusc_infos:
                for name in set(info["gt_names"]):
                    if name in self.meta["class_names"]:
                        _cls_infos[name].append(info)
            _cls_dist = {k: len(v) / len(_nusc_infos) for k, v in _cls_infos.items()}
            self.meta["sample_dist"] = _cls_dist
        else:
            if isinstance(_nusc_infos_all, dict):
                _nusc_infos = []
                for v in _nusc_infos_all.values():
                    _nusc_infos.extend(v)
            else:
                _nusc_infos = _nusc_infos_all

        return _nusc_infos

    def __len__(self):
        return len(self.dataset_dicts)

    def __getitem__(self, idx):
        info = deepcopy(self.dataset_dicts[idx])

        if info["lidar_path"].startswith("datasets/nuscenes"):
            lidar_path = os.path.join(os.environ["EFG_PATH"], info["lidar_path"])
        points = read_file(lidar_path)

        # points[:, 3] /= 255
        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        nsweeps = self.meta["nsweeps"]
        assert (nsweeps - 1) <= len(info["sweeps"]), "nsweeps {} should not greater than list length {}.".format(
            nsweeps, len(info["sweeps"])
        )

        for i in range(nsweeps - 1):
            sweep = info["sweeps"][i]
            if sweep["lidar_path"].startswith("datasets/nuscenes"):
                slidar_path = os.path.join(os.environ["EFG_PATH"], sweep["lidar_path"])
            sweep["lidar_path"] = slidar_path

            points_sweep, times_sweep = read_sweep(sweep)
            if points_sweep is None or times_sweep is None:
                continue
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        points = np.hstack([points, times])
        # convert from nuscenes coordinates to EFG standard coordinates
        # x, y -> y, -x
        points[:, :2] = points[:, [1, 0]]
        points[:, 1] *= -1

        # prepare annotations
        info["metadata"] = {
            "root_path": self.root_path,
            "db_path": self.db_path,
            "num_point_features": points.shape[-1],
        }

        if self.is_train:
            # N x 9: [x, y, z, l, w, h, vx, vy, r]
            mask = drop_arrays_by_name(
                info["gt_names"],
                [
                    "ignore",
                ],
            )
            info["annotations"] = {
                "gt_boxes": info.pop("gt_boxes")[mask],
                "gt_names": info.pop("gt_names")[mask],
                "tokens": info.pop("gt_boxes_token")[mask],
            }

        info["root_path"] = self.root_path

        points, info = self._apply_transforms(points, info)

        if self.is_train:
            self._filter_gt_by_classes(info)
            self._add_class_labels_to_annos(info)

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
