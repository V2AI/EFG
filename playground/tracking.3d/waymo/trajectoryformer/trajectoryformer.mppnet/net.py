from aug import *
from sample import *
from track_evaluator import *
from env import *
from trajectoryformer import TrajectoryFormer


def build_model(self, config):
    model = TrajectoryFormer(config).cuda()

    return model
