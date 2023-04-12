from .registry import TRAINERS
from .trainer import *


def build_trainer(config, build_model):
    trainer_class = TRAINERS.get(config.trainer.type)
    trainer_class.build_model = classmethod(build_model)
    return trainer_class(config)
