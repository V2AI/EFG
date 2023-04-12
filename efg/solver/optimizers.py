from typing import Any, Dict, List, Set

import torch
from torch import optim

from .registry import OPTIMIZERS

NORM_MODULE_TYPES = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
    # NaiveSyncBatchNorm inherits from BatchNorm2d
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.LayerNorm,
    torch.nn.LocalResponseNorm,
)


@OPTIMIZERS.register()
class Adam:
    @staticmethod
    def build(config, model):
        optim_config = config.solver.optimizer
        optimizer = torch.optim.Adam(model.parameters(), **optim_config)
        return optimizer


@OPTIMIZERS.register()
class AdamW:
    @staticmethod
    def build(config, model):
        optim_config = config.solver.optimizer
        optimizer = torch.optim.AdamW(model.parameters(), **optim_config)
        return optimizer


@OPTIMIZERS.register()
class D2_SGD:
    @staticmethod
    def build(config, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module in model.modules():
            for key, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                lr = config.solver.optimizer.base_lr
                weight_decay = config.solver.optimizer.weight_decay
                if isinstance(module, NORM_MODULE_TYPES):
                    weight_decay = config.solver.optimizer.weight_decay_norm
                elif key == "bias":
                    # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                    # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                    # hyperparameters are by default exactly the same as for regular
                    # weights.
                    lr = config.solver.optimizer.base_lr * config.solver.optimizer.bias_lr_factor
                    weight_decay = config.solver.optimizer.weight_decay
                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        optimizer = optim.SGD(
            params,
            config.solver.optimizer.base_lr,
            momentum=config.solver.optimizer.momentum,
        )
        return optimizer
