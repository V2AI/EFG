import logging

import numpy as np

from efg.geometry.box_ops import (  # mask_boxes_outside_range,; mask_boxes_outside_range_bev,
    mask_boxes_outside_range_bev_z_bound,
    mask_boxes_outside_range_center,
    mask_points_by_range,
    mask_points_by_range_bev,
    points_in_rbbox,
    rotate_points_along_z
)

from .augmentations import AugmentationBase
from .db_sampler import DataBaseSampler
from .registry import PROCESSORS
from .voxel_generator import VoxelGenerator

logger = logging.getLogger(__name__)


def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            try:
                dict_[k] = v[inds]
            except IndexError:
                dict_[k] = v[inds[len(v)]]


@PROCESSORS.register()
class FilterByDifficulty(AugmentationBase):
    def __init__(self, filter_difficulties):
        super().__init__()
        self._init(locals())

    def _filter_annos(self, info):
        assert "annotations" in info
        anno_dict = info["annotations"]
        if "difficulty" in anno_dict:
            remove_mask = [d in self.filter_difficulties for d in anno_dict["difficulty"]]
            keep_mask = np.logical_not(remove_mask)
            _dict_select(anno_dict, keep_mask)
        return info

    def __call__(self, points, info):
        if "annotations" in info:
            info = self._filter_annos(info)
            for sweep in info["sweeps"]:
                if "annotations" in sweep:
                    sweep = self._filter_annos(sweep)
        return points, info


@PROCESSORS.register()
class DatabaseSampling(AugmentationBase):
    def __init__(self, db_info_path, sample_groups, min_points=0, difficulty=-1, p=1.0, rm_points_after_sample=False):
        super().__init__()

        self.p = p
        self.rm_points_after_sample = rm_points_after_sample
        self.db_sampler = self.build_database_sampler(db_info_path, sample_groups, min_points, difficulty)

    def build_database_sampler(self, db_info_path, sample_groups, min_points, difficulty):
        db_sampler = DataBaseSampler(
            db_info_path,
            sample_groups,
            min_points=min_points,
            difficulty=difficulty,
        )
        return db_sampler

    def __call__(self, points, info):
        if self._rand_range() <= self.p:
            sampled_dict = self.db_sampler.sample_all(
                info["metadata"]["db_path"],
                info["annotations"]["gt_boxes"],
                info["annotations"]["gt_names"],
                info["metadata"]["num_point_features"],
            )

            if sampled_dict is not None:
                for k in ["gt_names", "gt_boxes"]:
                    info["annotations"][k] = np.concatenate([info["annotations"][k], sampled_dict[k]], axis=0)
                for k in ["difficulty", "num_points_in_gt"]:
                    if k in info["annotations"]:
                        info["annotations"][k] = np.concatenate([info["annotations"][k], sampled_dict[k]], axis=0)
                info["annotations"]["gt_boxes"] = np.nan_to_num(info["annotations"]["gt_boxes"])

                if self.rm_points_after_sample:
                    sampled_gt_boxes = np.nan_to_num(sampled_dict["gt_boxes"])
                    masks = points_in_rbbox(points, sampled_gt_boxes)
                    points = points[np.logical_not(masks.any(-1))]

                sampled_points = sampled_dict["points"]
                points = np.nan_to_num(np.concatenate([sampled_points, points], axis=0))

        return points, info


@PROCESSORS.register()
class DatabaseSamplingSim(DatabaseSampling):
    def build_database_sampler(self, db_info_path, sample_groups, min_points, difficulty):
        db_sampler = DataBaseSampler(
            db_info_path,
            sample_groups,
            min_points=min_points,
            difficulty=difficulty,
            sample_func="rand_sample",
        )
        return db_sampler


@PROCESSORS.register()
class PointShuffle(AugmentationBase):
    def __init__(self, p=0.5):
        super().__init__()
        self._init(locals())

    def __call__(self, points, info):
        if self._rand_range() <= self.p:
            np.random.shuffle(points)
        return points, info


@PROCESSORS.register()
class RandomFlip3D(AugmentationBase):
    def __init__(self, p=0.5):
        super().__init__()
        self._init(locals())

    def __call__(self, points, info):
        # flip along x axis
        if np.random.choice([False, True], replace=False, p=[1 - self.p, self.p]):
            points[:, 1] = -points[:, 1]
            if "annotations" in info:
                gt_boxes = info["annotations"]["gt_boxes"]
                gt_boxes[:, 1] = -gt_boxes[:, 1]
                gt_boxes[:, -1] = -gt_boxes[:, -1]
                # flip velocity_y
                if gt_boxes.shape[1] > 7:
                    gt_boxes[:, 7] = -gt_boxes[:, 7]
                for sweep in info["sweeps"]:
                    if "annotations" in sweep:
                        sgt_boxes = sweep["annotations"]["gt_boxes"]
                        sgt_boxes[:, 1] = -sgt_boxes[:, 1]
                        sgt_boxes[:, -1] = -sgt_boxes[:, -1]
                        # flip velocity_y
                        if sgt_boxes.shape[1] > 7:
                            sgt_boxes[:, 7] = -sgt_boxes[:, 7]
        # flip along y axis
        if np.random.choice([False, True], replace=False, p=[1 - self.p, self.p]):
            points[:, 0] = -points[:, 0]
            if "annotations" in info:
                gt_boxes = info["annotations"]["gt_boxes"]
                gt_boxes[:, 0] = -gt_boxes[:, 0]
                gt_boxes[:, -1] = -(gt_boxes[:, -1] + np.pi)
                # flip velocity_x
                if gt_boxes.shape[1] > 7:  # x, y, z, l, w, h, vx, vy, yaw
                    gt_boxes[:, 6] = -gt_boxes[:, 6]
                for sweep in info["sweeps"]:
                    if "annotations" in sweep:
                        sgt_boxes = sweep["annotations"]["gt_boxes"]
                        sgt_boxes[:, 0] = -sgt_boxes[:, 0]
                        sgt_boxes[:, -1] = -(sgt_boxes[:, -1] + np.pi)
                        # flip velocity_x
                        if sgt_boxes.shape[1] > 7:
                            sgt_boxes[:, 6] = -sgt_boxes[:, 6]
        return points, info


@PROCESSORS.register()
class GlobalRotation(AugmentationBase):
    def __init__(self, rotation):
        super().__init__()
        if not isinstance(rotation, list):
            rotation = [-rotation, rotation]
        self._init(locals())

    def _rotate_annos(self, info, noise_rotation):
        assert "annotations" in info
        target = info["annotations"]
        target["gt_boxes"][:, :3] = rotate_points_along_z(
            target["gt_boxes"][np.newaxis, :, :3], np.array([noise_rotation])
        )[0]
        target["gt_boxes"][:, -1] += noise_rotation
        if target["gt_boxes"].shape[1] > 7:
            target["gt_boxes"][:, 6:8] = rotate_points_along_z(
                np.hstack([target["gt_boxes"][:, 6:8], np.zeros((target["gt_boxes"].shape[0], 1))])[np.newaxis, :, :],
                np.array([noise_rotation]),
            )[0, :, :2]
        return info

    def __call__(self, points, info):
        noise_rotation = np.random.uniform(self.rotation[0], self.rotation[1])
        points = rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
        if "annotations" in info:
            info = self._rotate_annos(info, noise_rotation)
            for sweep in info["sweeps"]:
                if "annotations" in sweep:
                    sweep = self._rotate_annos(sweep, noise_rotation)
        return points, info


@PROCESSORS.register()
class GlobalScaling(AugmentationBase):
    def __init__(self, min_scale, max_scale):
        super().__init__()
        self._init(locals())

    def __call__(self, points, info):
        noise_scale = np.random.uniform(self.min_scale, self.max_scale)
        points[:, :3] *= noise_scale
        if "annotations" in info:
            gt_boxes = info["annotations"]["gt_boxes"]
            gt_boxes[:, :-1] *= noise_scale
            for sweep in info["sweeps"]:
                if "annotations" in sweep:
                    sweep["annotations"]["gt_boxes"][:, :-1] *= noise_scale
        return points, info


@PROCESSORS.register()
class GlobalTranslation(AugmentationBase):
    def __init__(self, std=[0, 0, 0]):
        super()._init(locals())

    def __call__(self, points, info):
        translation_std = np.array(self.std, dtype=np.float32)
        trans_vector = np.random.normal(scale=translation_std, size=3).T
        points[:, :3] += trans_vector
        if "annotations" in info:
            gt_boxes = info["annotations"]["gt_boxes"]
            gt_boxes[:, :3] += trans_vector
            for sweep in info["sweeps"]:
                if "annotations" in sweep:
                    sweep["annotations"]["gt_boxes"][:, :3] += trans_vector
        return points, info


@PROCESSORS.register()
class PointsJitter(AugmentationBase):
    def __init__(self, jitter_std=[0.01, 0.01, 0.01], clip_range=[-0.05, 0.05]):
        super()._init(locals())

    def __call__(self, points, info):
        jitter_std = np.array(self.jitter_std, dtype=np.float32)
        jitter_noise = np.random.randn(points.shape[0], 3) * jitter_std[None, :]
        if self.clip_range is not None:
            jitter_noise = np.clip(jitter_noise, self.clip_range[0], self.clip_range[1])
        points[:, :3] += jitter_noise

        return points, info


@PROCESSORS.register()
class Voxelization(AugmentationBase):
    def __init__(self, pc_range, voxel_size, max_points_in_voxel, max_voxel_num):
        super().__init__()
        self._init(locals())
        self.voxel_generator = VoxelGenerator(
            voxel_size=voxel_size,
            point_cloud_range=pc_range,
            max_num_points=max_points_in_voxel,
            max_voxels=max_voxel_num,
        )

    def __call__(self, points, info):
        # [0, -40, -3, 70.4, 40, 1]
        pc_range = self.voxel_generator.point_cloud_range
        grid_size = self.voxel_generator.grid_size
        voxels, coordinates, num_points_per_voxel = self.voxel_generator.generate(points)
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
        point_voxels = dict(
            voxels=voxels,
            points=points,
            coordinates=coordinates,
            num_points_per_voxel=num_points_per_voxel,
            num_voxels=num_voxels,
            shape=grid_size,
            range=pc_range,
            size=self.voxel_size,
        )
        return point_voxels, info


@PROCESSORS.register()
class FilterByRange(AugmentationBase):
    def __init__(self, pc_range, with_gt=True):
        super().__init__()
        pc_range = np.array(list(pc_range))
        self._init(locals())
        self._set_filter_func()

    def _set_filter_func(self):
        self.filter_func = mask_boxes_outside_range_bev_z_bound

    def _filter_annos(self, info):
        assert "annotations" in info
        target = info["annotations"]
        keep = self.filter_func(target["gt_boxes"], self.pc_range)
        _dict_select(target, keep)
        return info

    def __call__(self, points, info):
        keep = mask_points_by_range(points, self.pc_range)
        points = points[keep]
        if self.with_gt:
            if "annotations" in info:
                info = self._filter_annos(info)
                for sweep in info["sweeps"]:
                    if "annotations" in sweep:
                        sweep = self._filter_annos(sweep)
        return points, info


@PROCESSORS.register()
class FilterByRangeCenter(FilterByRange):
    def _set_filter_func(self):
        self.filter_func = mask_boxes_outside_range_center


@PROCESSORS.register()
class FilterByRangeXY(FilterByRange):
    def _set_filter_func(self):
        self.filter_func = mask_points_by_range_bev


@PROCESSORS.register()
class RandomCropPoints(AugmentationBase):
    def __init__(self, crop_type, crop_size, pc_range, p=0.5):
        """
        Args:
            crop_type (str): one of "relative_range", "relative", "absolute", "absolute_range".
            crop_size (tuple[float, float]): two floats, explained below.
        - "relative": crop a (H * crop_size[0], W * crop_size[1]) region from an input image of
          size (H, W). crop size should be in (0, 1]
        - "relative_range": uniformly sample two values from [crop_size[0], 1]
          and [crop_size[1]], 1], and use them as in "relative" crop type.
        - "absolute" crop a (crop_size[0], crop_size[1]) region from input image.
          crop_size must be smaller than the input image size.
        - "absolute_range", for an input of size (H, W), uniformly sample H_crop in
          [crop_size[0], min(H, crop_size[1])] and W_crop in [crop_size[0], min(W, crop_size[1])].
          Then crop a region (H_crop, W_crop).
        """
        # TODO style of relative_range and absolute_range are not consistent:
        # one takes (h, w) but another takes (min, max)
        super().__init__()
        assert crop_type in ["relative_range", "relative", "absolute", "absolute_range"]
        self._init(locals())

    def get_crop_size(self, shapes):
        """
        Args:
            image_size (tuple): height, width
        Returns:
            crop_size (tuple): height, width in absolute pixels
        """

        h, w = shapes
        assert h == w, "Only support the same crop size along H and W."

        if self.crop_type == "relative":
            ch, cw = self.crop_size
            # return h * ch, w * cw
            return h * ch, w * ch
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            # return h * ch, w * cw
            return h * ch, w * ch
        elif self.crop_type == "absolute":
            # return (min(self.crop_size[0], h), min(self.crop_size[1], w))
            return (min(self.crop_size[0], h), min(self.crop_size[0], h))
        elif self.crop_type == "absolute_range":
            assert self.crop_size[0] <= self.crop_size[1]
            ch = np.random.rand(min(h, self.crop_size[0]), min(h, self.crop_size[1]))
            # cw = np.random.rand(min(w, self.crop_size[0]), min(w, self.crop_size[1]))
            # return ch, cw
            return ch, ch
        else:
            raise NotImplementedError("Unknown crop type {}".format(self.crop_type))

    def __call__(self, points, info):
        if self._rand_range() <= self.p:
            h, w = np.array(self.pc_range[3:5]) - np.array(self.pc_range[:2])

            self.h, self.w = self.get_crop_size((h, w))
            assert h >= self.h and w >= self.w, "Shape computation in {} has bugs.".format(self)

            self.x0 = np.random.randint(int(h - self.h) + 1) + self.h / 2
            self.y0 = np.random.randint(int(w - self.w) + 1) + self.w / 2

            center_offset = np.array([self.x0 - h / 2, self.y0 - w / 2])

            if "annotations" in info:
                gt_boxes = info["annotations"]["gt_boxes"]
                info["annotations"]["gt_boxes"] = self.apply_coords(info["annotations"]["gt_boxes"], center_offset)
                keep = mask_boxes_outside_range_bev_z_bound(
                    info["annotations"]["gt_boxes"],
                    np.array([-self.h / 2, -self.w / 2, -1000, self.h / 2, self.w / 2, 1000]),
                )
                _dict_select(info["annotations"], keep)

            points = self.apply_point_clouds(points, center_offset)

            # scale cropped point clouds up to original size
            h_scale = h / self.h
            w_scale = w / self.w

            points[:, 0] *= h_scale
            points[:, 1] *= w_scale

            if "annotations" in info:
                gt_boxes = info["annotations"]["gt_boxes"]
                gt_boxes[:, [0, 3]] *= h_scale
                gt_boxes[:, [1, 4]] *= w_scale

                if gt_boxes.shape[1] == 9:
                    gt_boxes[:, 6] *= h_scale
                    gt_boxes[:, 7] *= w_scale

                # rotate angle alone h_scle and w_scle

        return points, info

    def apply_point_clouds(self, points: np.ndarray, center_offset: np.ndarray) -> np.ndarray:
        points[..., :2] -= np.array(self.pc_range[:2])
        crop_range = [self.x0 - self.h / 2, self.y0 - self.w / 2, self.x0 + self.h / 2, self.y0 + self.w / 2]
        mask = (
            (points[:, 0] > crop_range[0])
            & (points[:, 0] < crop_range[2])
            & (points[:, 1] > crop_range[1])
            & (points[:, 1] < crop_range[3])
        )
        cropped_points = points[mask]
        cropped_points[:, :2] = self.apply_coords(cropped_points[:, :2] + np.array(self.pc_range[:2]), center_offset)

        return cropped_points

    def apply_coords(self, coords: np.ndarray, center_offset: np.ndarray) -> np.ndarray:
        coords[:, :2] -= center_offset
        return coords
