from sample import *
from track_evaluator import *
from env import *
from modules.optimizer import *
from trajectoryformer import TrajectoryFormer


def build_model(self, config):

    model =  TrajectoryFormer(config).cuda()

    return model
