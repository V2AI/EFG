from aug import *
from sample import *
from env import *
from modules.optimizer import *
from motionpred import MotionPrediction

def build_model(self, config):

    model =  MotionPrediction(config)

    return model
