from .lr_schedulers import *
from .optimizers import *
from .registry import LR_SCHEDULERS, OPTIMIZERS


def build_optimizer(config, model):
    optim_type = config.solver.optimizer.pop("type")  # the rest key-values pairs are args
    optimizer = OPTIMIZERS.get(optim_type).build(config, model)
    config.solver.optimizer.type = optim_type

    return optimizer


def build_scheduler(config, optimizer):
    scheduler_type = config.solver.lr_scheduler.pop("type")
    scheduler = LR_SCHEDULERS.get(scheduler_type).build(config, optimizer)
    config.solver.lr_scheduler.type = scheduler_type

    return scheduler
