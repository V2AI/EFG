import copy
import logging
import math
import os
import pickle

import numpy as np

from efg.geometry.box_ops import box_collision_test, center_to_corner_box2d
from efg.utils.distributed import get_rank, get_world_size, is_dist_avail_and_initialized

logger = logging.getLogger(__name__)


class BatchSampler:
    def __init__(self, sampled_list, name=None, shuffle=True):
        num_replicas = 1
        rank = 0

        if is_dist_avail_and_initialized():
            num_replicas = get_world_size()
            rank = get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(sampled_list) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        self._sampled_list = sampled_list
        self._indices = self._get_indices(shuffle)

        self._idx = 0
        self._name = name
        self._shuffle = shuffle

    def _get_indices(self, shuffle):
        indices = np.arange(len(self._sampled_list)).tolist()
        if shuffle:
            np.random.shuffle(indices)
        indices += indices[: (self.total_size - len(self._sampled_list))]
        assert len(indices) == self.total_size

        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples

        return indices

    def _sample(self, num):
        if self._idx + num >= self.num_samples:
            ret = self._indices[self._idx :].copy()
            self._reset()
        else:
            ret = self._indices[self._idx : self._idx + num]
            self._idx += num
        return ret

    def _reset(self):
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0

    def sample(self, num):
        indices = self._sample(num)
        return [self._sampled_list[i] for i in indices]


class DataBaseSampler:
    def __init__(self, db_info_path, groups, min_points=0, difficulty=-1, sample_func="sample"):
        self.db_info_path = db_info_path
        self.min_points = min_points
        self.difficulty = difficulty
        self.sample_func = getattr(self, sample_func)

        self._groups = groups
        self._group_db_infos = {}
        self._group_name_to_names = []
        self._sample_classes = []
        self._sample_max_nums = []

        self.init_db_infos()

    def init_db_infos(
        self,
    ):
        with open(self.db_info_path, "rb") as f:
            db_infos = pickle.load(f)

        logger.info(f"Filtering database with difficulty {self.difficulty} and min_points {self.min_points}")
        new_db_infos = {}
        for name, db_info in db_infos.items():
            new_db_infos[name] = [
                info
                for info in db_info
                if info["num_points_in_gt"] >= self.min_points and info["difficulty"] >= self.difficulty
            ]
            logger.info(f"Loading {len(db_info)} -> {len(new_db_infos[name])} {name} database infos.")
        db_infos = new_db_infos

        # configure group samplers
        self._group_db_infos = db_infos  # just use db_infos
        for group_info in self._groups:
            group_names = list(group_info.keys())
            self._sample_classes += group_names
            self._sample_max_nums += list(group_info.values())
        self._sampler_dict = {}
        for k, v in self._group_db_infos.items():
            self._sampler_dict[k] = BatchSampler(v, k)

    def sample_all(self, root_path, gt_boxes, gt_names, num_point_features):
        sampled_num_dict = {}
        sample_num_per_class = []

        for class_name, max_sample_num in zip(self._sample_classes, self._sample_max_nums):
            sampled_num = int(max_sample_num - np.sum([n == class_name for n in gt_names]))
            sampled_num = np.round(sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled_groups = self._sample_classes
        sampled = []
        sampled_gt_boxes = []
        avoid_coll_boxes = gt_boxes

        for class_name, sampled_num in zip(sampled_groups, sample_num_per_class):
            if sampled_num > 0:
                sampled_cls = self.sample_class(class_name, sampled_num, avoid_coll_boxes)

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]["box3d_lidar"][np.newaxis, ...]
                    else:
                        sampled_gt_box = np.stack([s["box3d_lidar"] for s in sampled_cls], axis=0)

                    sampled_gt_boxes += [sampled_gt_box]
                    avoid_coll_boxes = np.concatenate([avoid_coll_boxes, sampled_gt_box], axis=0)

        if len(sampled) > 0:
            sampled_gt_boxes = np.concatenate(sampled_gt_boxes, axis=0)

            return_gt_boxes = []
            return_sampled = []
            s_points_list = []
            for info, gt_box in zip(sampled, sampled_gt_boxes):
                info_path = os.path.join(root_path, info["path"])
                s_points = np.fromfile(info_path, dtype=np.float32).reshape(-1, num_point_features)
                s_points[:, :3] += info["box3d_lidar"][:3]
                s_points_list.append(s_points)
                return_sampled.append(info)
                return_gt_boxes.append(gt_box)

            if len(s_points_list) == 0:
                return None

            ret = {
                "gt_boxes": np.array(return_gt_boxes),
                "gt_names": np.array([s["name"] for s in return_sampled]),
                "difficulty": np.array([s["difficulty"] for s in return_sampled]),
                "num_points_in_gt": np.array([s["num_points_in_gt"] for s in return_sampled]),
                "points": np.concatenate(s_points_list, axis=0),
            }
        else:
            ret = None

        return ret

    def sample(self, name, num):
        ret = self._sampler_dict[name].sample(num)
        return ret

    def rand_sample(self, name, num):
        ret = copy.deepcopy(np.random.choice(self._group_db_infos[name], num))
        return ret

    def sample_class(self, name, num, gt_boxes):
        sampled = copy.deepcopy(self.sample_func(name, num))
        num_gt = gt_boxes.shape[0]
        num_sampled = len(sampled)
        gt_boxes_bv = center_to_corner_box2d(gt_boxes[:, 0:2], gt_boxes[:, 3:5], gt_boxes[:, -1])

        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0)

        valid_mask = np.zeros([gt_boxes.shape[0]], dtype=np.bool_)
        valid_mask = np.concatenate([valid_mask, np.ones([sp_boxes.shape[0]], dtype=np.bool_)], axis=0)
        boxes = np.concatenate([gt_boxes, sp_boxes], axis=0).copy()

        sp_boxes_new = boxes[gt_boxes.shape[0] :]
        sp_boxes_bv = center_to_corner_box2d(sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, -1])

        total_bv = np.concatenate([gt_boxes_bv, sp_boxes_bv], axis=0)
        coll_mat = box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])

        return valid_samples
