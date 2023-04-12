from augmentations import FilterByRangeInPolygon
from voxelnet import VoxelNet

__all__ = ["build_model", "FilterByRangeInPolygon"]


def build_model(self, cfg):
    model = VoxelNet(cfg)

    return model
