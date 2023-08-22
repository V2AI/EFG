import logging
import numpy as np
from efg.geometry.box_ops import rotate_points_along_z
from efg.data.augmentations import AugmentationBase
from efg.data.registry import PROCESSORS


logger = logging.getLogger(__name__)


def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if "pred" in k or "future" in k:
            continue
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            try:
                dict_[k] = v[inds]
            except IndexError:
                dict_[k] = v[inds[len(v)]]


@PROCESSORS.register()
class RandomFlip3DFutureGT(AugmentationBase):
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

                future_gt_boxes = info["annotations"]["future_gt_boxes"]
                future_gt_boxes[:, 1] = -future_gt_boxes[:, 1]
                future_gt_boxes[:, -1] = -future_gt_boxes[:, -1]

                if "pred_boxes3d" in info["annotations"].keys():
                    pred_boxes3d = info["annotations"]["pred_boxes3d"]
                    pred_boxes3d[:, 1] = -pred_boxes3d[:, 1]
                    pred_boxes3d[:, -1] = -pred_boxes3d[:, -1]
                    pred_boxes3d[:, 7] = -pred_boxes3d[:, 7]

                # flip velocity_y
                if gt_boxes.shape[1] > 7:
                    gt_boxes[:, 7] = -gt_boxes[:, 7]
                    future_gt_boxes[:, 7] = -future_gt_boxes[:, 7]
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

                future_gt_boxes = info["annotations"]["future_gt_boxes"]
                future_gt_boxes[:, 0] = -future_gt_boxes[:, 0]
                future_gt_boxes[:, -1] = -(future_gt_boxes[:, -1] + np.pi)

                if "pred_boxes3d" in info["annotations"].keys():
                    pred_boxes3d = info["annotations"]["pred_boxes3d"]
                    pred_boxes3d[:, 0] = -pred_boxes3d[:, 0]
                    pred_boxes3d[:, -1] = -(pred_boxes3d[:, -1] + np.pi)
                    pred_boxes3d[:, 6] = -pred_boxes3d[:, 6]

                # flip velocity_x
                if gt_boxes.shape[1] > 7:
                    gt_boxes[:, 6] = -gt_boxes[:, 6]
                    future_gt_boxes[:, 6] = -future_gt_boxes[:, 6]
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
class GlobalRotationFutureGT(AugmentationBase):
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

        if "future_gt_boxes" in target.keys():
            target["future_gt_boxes"][:, :3] = rotate_points_along_z(
                target["future_gt_boxes"][np.newaxis, :, :3], np.array([noise_rotation])
            )[0]
            target["future_gt_boxes"][:, -1] += noise_rotation

        if "pred_boxes3d" in target.keys():
            target["pred_boxes3d"][:, :3] = rotate_points_along_z(
                target["pred_boxes3d"][np.newaxis, :, :3], np.array([noise_rotation])
            )[0]
            target["pred_boxes3d"][:, -1] += noise_rotation

            if target["pred_boxes3d"].shape[1] > 7:
                target["pred_boxes3d"][:, 6:8] = rotate_points_along_z(
                    np.hstack(
                        [
                            target["pred_boxes3d"][:, 6:8],
                            np.zeros((target["pred_boxes3d"].shape[0], 1)),
                        ]
                    )[np.newaxis, :, :],
                    np.array([noise_rotation]),
                )[0, :, :2]

        if target["gt_boxes"].shape[1] > 7:
            target["gt_boxes"][:, 6:8] = rotate_points_along_z(
                np.hstack(
                    [
                        target["gt_boxes"][:, 6:8],
                        np.zeros((target["gt_boxes"].shape[0], 1)),
                    ]
                )[np.newaxis, :, :],
                np.array([noise_rotation]),
            )[0, :, :2]

            if "future_gt_boxes" in target.keys():
                target["future_gt_boxes"][:, 6:8] = rotate_points_along_z(
                    np.hstack(
                        [
                            target["future_gt_boxes"][:, 6:8],
                            np.zeros((target["future_gt_boxes"].shape[0], 1)),
                        ]
                    )[np.newaxis, :, :],
                    np.array([noise_rotation]),
                )[0, :, :2]

        return info

    def __call__(self, points, info):
        noise_rotation = np.random.uniform(self.rotation[0], self.rotation[1])
        points = rotate_points_along_z(
            points[np.newaxis, :, :], np.array([noise_rotation])
        )[0]
        if "annotations" in info:
            info = self._rotate_annos(info, noise_rotation)
            for sweep in info["sweeps"]:
                if "annotations" in sweep:
                    sweep = self._rotate_annos(sweep, noise_rotation)
        return points, info


@PROCESSORS.register()
class GlobalScalingFutureGT(AugmentationBase):
    def __init__(self, min_scale, max_scale):
        super().__init__()
        self._init(locals())

    def __call__(self, points, info):
        noise_scale = np.random.uniform(self.min_scale, self.max_scale)
        points[:, :3] *= noise_scale
        if "annotations" in info:
            gt_boxes = info["annotations"]["gt_boxes"]
            gt_boxes[:, :-1] *= noise_scale

            future_gt_boxes = info["annotations"]["future_gt_boxes"]
            future_gt_boxes[:, :-1] *= noise_scale

            if "pred_boxes3d" in info["annotations"].keys():
                pred_boxes3d = info["annotations"]["pred_boxes3d"]
                pred_boxes3d[:, :-1] *= noise_scale

            for sweep in info["sweeps"]:
                if "annotations" in sweep:
                    sweep["annotations"]["gt_boxes"][:, :-1] *= noise_scale
        return points, info
