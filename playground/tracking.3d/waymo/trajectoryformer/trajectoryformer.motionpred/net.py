from aug import RandomFlip3DFutureGT, GlobalRotationFutureGT, GlobalScalingFutureGT
from sample import SeqInferenceSampler
from env import CustomWDDataset
from motionpred import MotionPrediction


__all__ = [
    "RandomFlip3DFutureGT", "GlobalRotationFutureGT", "GlobalScalingFutureGT",
    "SeqInferenceSampler", "CustomWDDataset"
]


def build_model(self, config):
    model = MotionPrediction(config)
    return model
