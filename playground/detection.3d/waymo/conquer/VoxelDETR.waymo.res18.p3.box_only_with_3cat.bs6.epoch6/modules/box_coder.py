from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np

import torch

from efg.geometry.box_ops_torch import limit_period

from .utils import normalize_period


class BoxCoder:
    __metaclass__ = ABCMeta

    @abstractproperty
    def code_size(self):
        pass

    def encode(self, boxes, **kwargs):
        return self._encode(boxes, **kwargs)

    def decode(self, rel_codes, **kwargs):
        return self._decode(rel_codes, **kwargs)

    @abstractmethod
    def _encode(self, boxes, **kwargs):
        pass

    @abstractmethod
    def _decode(self, rel_codes, **kwargs):
        pass


class VoxelBoxCoder3D(BoxCoder):
    def __init__(self, voxel_size, pc_range, n_dim=7, device=torch.device("cpu"), **opts):
        self.device = device
        self.voxel_size = torch.tensor(voxel_size, device=device)
        self.pc_range = torch.tensor(pc_range, device=device)
        self.pc_size = self.pc_range[3:] - self.pc_range[:3]
        self.z_normalizer = 10.0
        self.grid_size = self.pc_size.div(self.voxel_size, rounding_mode="trunc")
        self.n_dim = n_dim
        for k, v in opts.items():
            setattr(self, k, v)

    @property
    def code_size(self):
        return self.n_dim

    def _encode(self, target):
        target["labels"] -= 1

        target["gt_boxes"][:, :2] -= self.pc_range[:2]
        target["gt_boxes"][:, :2] /= self.pc_size[:2]

        target["gt_boxes"][:, 2] -= -1 * self.z_normalizer  # -10 ~ 10
        target["gt_boxes"][:, 2] /= 2 * self.z_normalizer

        target["gt_boxes"][:, 3:5] /= self.pc_size[:2]
        target["gt_boxes"][:, 5] /= 2 * self.z_normalizer

        target["gt_boxes"][:, -1] = limit_period(target["gt_boxes"][:, -1], offset=0.5, period=np.pi * 2)

        # TODO: temporarily remove velocity, need to add when move to tracking
        target["gt_boxes"] = target["gt_boxes"][:, [0, 1, 2, 3, 4, 5, -1]]
        target["gt_boxes"][:, -1] = normalize_period(target["gt_boxes"][:, -1], offset=0.5, period=np.pi * 2)

        assert ((target["gt_boxes"] >= 0) & (target["gt_boxes"] <= 1)).all().item()

        return target

    def _decode(self, pred_boxes):
        # recover predictions
        pred_boxes[..., :2] = pred_boxes[..., :2] * self.pc_size[:2] + self.pc_range[:2]
        pred_boxes[..., 2] = pred_boxes[..., 2] * 2 * self.z_normalizer + -1 * self.z_normalizer
        pred_boxes[..., 3:5] = pred_boxes[..., 3:5] * self.pc_size[:2]
        pred_boxes[..., 5] = pred_boxes[..., 5] * 2 * self.z_normalizer
        pred_boxes[..., -1] = pred_boxes[..., -1] * np.pi * 2 - np.pi

        return pred_boxes
