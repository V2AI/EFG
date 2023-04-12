from env import CustomTrainer, CustomWDDataset
from voxelnet import VoxelNet

__all__ = ["build_model", "CustomWDDataset", "CustomTrainer"]


def build_model(self, cfg):
    model = VoxelNet(cfg)

    return model
