from env import CustomTrainer, CustomWDDataset
from modules.optimizer import AdamWMulti
from voxel_detr import VoxelDETR

__all__ = ["build_model", "AdamWMulti", "CustomTrainer", "CustomWDDataset", "VoxelDETR"]


def build_model(self, config):
    model = VoxelDETR(config)

    return model
