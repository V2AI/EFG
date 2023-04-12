import math
import warnings
from bisect import bisect_right
from typing import List

import torch
from torch.optim.lr_scheduler import OneCycleLR, _LRScheduler

from .registry import LR_SCHEDULERS


def _get_warmup_factor_at_iter(method: str, iter: int, warmup_iters: int, warmup_factor: float) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See https://arxiv.org/abs/1706.02677 for more details.
    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).
    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    elif method == "burnin":
        return (iter / warmup_iters) ** 4
    else:
        raise ValueError("Unknown warmup method: {}".format(method))


class WarmupMultiStepLR(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        """
        Multi Step LR with warmup

        Args:
            optimizer (torch.optim.Optimizer): optimizer used.
            milestones (list[Int]): a list of increasing integers.
            gamma (float): gamma
            warmup_factor (float): lr = warmup_factor * base_lr
            warmup_iters (int): iters to warmup
            warmup_method (str): warmup method in ["constant", "linear", "burnin"]
            last_epoch(int):  The index of last epoch. Default: -1.
        """
        if not list(milestones) == sorted(milestones):
            raise ValueError(f"Milestones should be a list of increasing integers. Got {milestones}.")

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


class WarmupCosineAnnealingLR(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(
        self,
        optimizer,
        max_iters,
        warmup_iters=0,
        warmup_method="linear",
        warmup_factor=0.001,
        eta_min=0,
        last_epoch=-1,
        verbose=False,
    ):
        self.T_max = max_iters
        self.T_warmup = warmup_iters
        self.warmup_method = warmup_method
        self.warmup_factor = warmup_factor
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.", UserWarning
            )

        if self.last_epoch == 0:
            return [self.warmup_factor * base_lr for base_lr in self.base_lrs]
        elif self.last_epoch < self.T_warmup:
            warmup_factor = _get_warmup_factor_at_iter(
                self.warmup_method, self.last_epoch, self.T_warmup, self.warmup_factor
            )
            return [
                group["lr"] + (base_lr - base_lr * warmup_factor) / (self.T_warmup - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif (self.last_epoch - 1 - self.T_max) % (2 * (self.T_max - self.T_warmup)) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.T_max - self.T_warmup))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.T_warmup) / (self.T_max - self.T_warmup)))
            / (1 + math.cos(math.pi * (self.last_epoch - self.T_warmup - 1) / (self.T_max - self.T_warmup)))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        if self.last_epoch < self.T_warmup:
            warmup_factor = _get_warmup_factor_at_iter(
                self.warmup_method, self.last_epoch, self.T_warmup, self.warmup_factor
            )

            return [
                base_lr * warmup_factor + self.last_epoch * (base_lr - base_lr * warmup_factor) / (self.T_warmup - 1)
                for base_lr in self.base_lrs
            ]
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.T_warmup) / (self.T_max - self.T_warmup)))
            / 2
            for base_lr in self.base_lrs
        ]


@LR_SCHEDULERS.register()
class WarmupCosineAnnealing:
    @staticmethod
    def build(config, optimizer):
        sconfig = config.solver.lr_scheduler
        max_epochs = sconfig.pop("max_epochs")
        epoch_iters = config.solver.lr_scheduler.pop("epoch_iters")
        lr_scheduler = WarmupCosineAnnealingLR(optimizer, **sconfig)
        sconfig.max_epochs = max_epochs
        config.solver.lr_scheduler.epoch_iters = epoch_iters
        return lr_scheduler


@LR_SCHEDULERS.register()
class OneCycle:
    @staticmethod
    def build(config, optimizer):
        max_lr = config.solver.optimizer.lr
        total_steps = config.solver.lr_scheduler.max_iters
        max_epochs = config.solver.lr_scheduler.pop("max_epochs")
        max_iters = config.solver.lr_scheduler.pop("max_iters")
        epoch_iters = config.solver.lr_scheduler.pop("epoch_iters")
        lr_scheduler = OneCycleLR(optimizer, max_lr, total_steps=total_steps, **config.solver.lr_scheduler)

        config.solver.lr_scheduler.max_epochs = max_epochs
        config.solver.lr_scheduler.max_iters = max_iters
        config.solver.lr_scheduler.epoch_iters = epoch_iters

        return lr_scheduler


@LR_SCHEDULERS.register()
class WarmupMultiStep:
    @staticmethod
    def build(config, optimizer):
        if "epoch_iters" in config.solver.lr_scheduler:
            epoch_iters = config.solver.lr_scheduler.epoch_iters
            steps = [epoch_iters * s for s in config.solver.lr_scheduler.steps]
        else:
            steps = config.solver.lr_scheduler.steps
        scheduler = WarmupMultiStepLR(
            optimizer,
            steps,
            config.solver.lr_scheduler.gamma,
            warmup_factor=config.solver.lr_scheduler.warmup_factor,
            warmup_iters=config.solver.lr_scheduler.warmup_iters,
            warmup_method=config.solver.lr_scheduler.warmup_method,
        )
        return scheduler
