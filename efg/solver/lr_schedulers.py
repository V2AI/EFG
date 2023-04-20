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


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Sets the learning rate of each parameter group to follow a linear warmup schedule between warmup_start_lr
    and base_lr followed by a cosine annealing schedule between base_lr and eta_min.

    .. warning::
        It is recommended to call :func:`.step()` for :class:`LinearWarmupCosineAnnealingLR`
        after each iteration as calling it after each epoch will keep the starting lr at
        warmup_start_lr for the first epoch which is 0 in most cases.

    .. warning::
        passing epoch to :func:`.step()` is being deprecated and comes with an EPOCH_DEPRECATION_WARNING.
        It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
        :func:`get_lr()`. Though this does not change the behavior of the scheduler, when passing
        epoch param to :func:`.step()`, the user should call the :func:`.step()` function before calling
        train and validation methods.

    Example:
        >>> layer = nn.Linear(10, 1)
        >>> optimizer = Adam(layer.parameters(), lr=0.02)
        >>> scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)
        >>> #
        >>> # the default case
        >>> for epoch in range(40):
        ...     # train(...)
        ...     # validate(...)
        ...     scheduler.step()
        >>> #
        >>> # passing epoch param case
        >>> for epoch in range(40):
        ...     scheduler.step(epoch)
        ...     # train(...)
        ...     # validate(...)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            / (
                1
                + math.cos(
                    math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]


@LR_SCHEDULERS.register()
class LinearWarmupCosineAnnealing:
    """
    config.solver.lr_scheduer
    Args:
       max_iters / max_epochs: int
       min_lr_ratio: float, default 0.001, multipier of config.solver.optimizer.lr
       warmup_iters: int, default 1000
       warmup_ratio: float, default 0.001
    """
    @staticmethod
    def build(config, optimizer):
        sconfig = config.solver.lr_scheduler
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            max_epochs=sconfig.max_iters,
            eta_min=config.solver.optimizer.lr * sconfig.min_lr_ratio,
            warmup_epochs=sconfig.warmup_iters,
            warmup_start_lr=config.solver.optimizer.lr * sconfig.warmup_ratio,
        )
        return scheduler


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
