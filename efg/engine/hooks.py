import datetime
import logging
import time
from collections import Counter, Mapping

from torch.nn.utils import clip_grad

from efg.utils import distributed as comm
from efg.utils.events import EventWriter
from efg.utils.timer import Timer

from .registry import HOOKS

logger = logging.getLogger(__name__)


def clip_grad_norm(params, **kwargs):
    return clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, params), **kwargs)


def clip_grad_value(params, **kwargs):
    clip_grad.clip_grad_value_(filter(lambda p: p.requires_grad, params), **kwargs)


def flatten_results_dict(results):
    """
    Expand a hierarchical dict of scalars into a flat dict of scalars.
    If results[k1][k2][k3] = v, the returned dict will have the entry
    {"k1/k2/k3": v}.
    Args:
        results (dict):
    """
    r = {}
    for k, v in results.items():
        if isinstance(v, Mapping):
            v = flatten_results_dict(v)
            for kk, vv in v.items():
                r[k + "/" + kk] = vv
        else:
            r[k] = v
    return r


class HookBase:
    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_step(self):
        pass

    def after_step(self):
        pass


@HOOKS.register()
class Optimization(HookBase):
    def __init__(self, grad_clipper=None):
        if grad_clipper.enabled:
            self.clip = True
            self.clip_type = grad_clipper.clip_type
            self.clip_params = grad_clipper.params
        else:
            self.clip = False

    def after_step(self):
        losses = self.trainer.outputs["losses"]

        self.trainer.optimizer.zero_grad()
        losses.backward()

        if self.clip:
            if self.clip_type == "norm":
                grad_norm = clip_grad_norm(self.trainer.model.parameters(), **self.clip_params)
                self.trainer.storage.put_scalar("grad_norm", grad_norm)
            elif self.clip_type == "value":
                clip_grad_value(self.trainer.model.parameters(), **self.clip_params)

        self.trainer.optimizer.step()


class LRScheduler(HookBase):
    """
    A hook which executes a torch builtin LR scheduler and summarizes the LR.
    It is executed after every iteration.
    """

    def __init__(self, optimizer, scheduler):
        """
        Args:
            optimizer (torch.optim.Optimizer):
            scheduler (torch.optim._LRScheduler)
        """
        self._optimizer = optimizer
        self._scheduler = scheduler

        # NOTE: some heuristics on what LR to summarize
        # summarize the param group with most parameters
        largest_group = max(len(g["params"]) for g in optimizer.param_groups)

        if largest_group == 1:
            # If all groups have one parameter,
            # then find the most common initial LR, and use it for summary
            lr_count = Counter([g["lr"] for g in optimizer.param_groups])
            lr = lr_count.most_common()[0][0]
            for i, g in enumerate(optimizer.param_groups):
                if g["lr"] == lr:
                    self._best_param_group_id = i
                    break
        else:
            for i, g in enumerate(optimizer.param_groups):
                if len(g["params"]) == largest_group:
                    self._best_param_group_id = i
                    break

    def after_step(self):
        lr = self.trainer.optimizer.param_groups[self._best_param_group_id]["lr"]
        self.trainer.storage.put_scalar("lr", lr, smoothing_hint=False)
        self.trainer.lr_scheduler.step()


@HOOKS.register()
class IterTimer(HookBase):
    def __init__(self, warmup_iter=5):
        self._warmup_iter = warmup_iter
        self._step_timer = Timer()

    def before_train(self):
        self._start_time = time.perf_counter()
        self._total_timer = Timer()
        self._total_timer.pause()

    def after_train(self):
        total_time = time.perf_counter() - self._start_time
        total_time_minus_hooks = self._total_timer.seconds()
        hook_time = total_time - total_time_minus_hooks

        num_iter = self.trainer.iter + 1 - self.trainer.start_iter - self._warmup_iter

        if num_iter > 0 and total_time_minus_hooks > 0:
            # Speed is meaningful only after warmup
            # NOTE this format is parsed by grep in some scripts
            logger.info(
                "Overall training speed: {} iterations in {} ({:.4f} s / it)".format(
                    num_iter,
                    str(datetime.timedelta(seconds=int(total_time_minus_hooks))),
                    total_time_minus_hooks / num_iter,
                )
            )

        logger.info(
            "Total training time: {} ({} on hooks)".format(
                str(datetime.timedelta(seconds=int(total_time))),
                str(datetime.timedelta(seconds=int(hook_time))),
            )
        )

    def before_step(self):
        self._step_timer.reset()
        self._total_timer.resume()

    def after_step(self):
        # +1 because we're in after_step
        iter_done = self.trainer.iter - self.trainer.start_iter + 1
        if iter_done >= self._warmup_iter:
            sec = self._step_timer.seconds()
            self.trainer.storage.put_scalars(time=sec)
        else:
            self._start_time = time.perf_counter()
            self._total_timer.reset()

        self._total_timer.pause()


@HOOKS.register()
class PeriodicWriter(HookBase):
    """
    Write events to EventStorage periodically.
    It is executed every ``period`` iterations and after the last iteration.
    """

    def __init__(self, writers, period=20):
        """
        Args:
            writers (list[EventWriter]): a list of EventWriter objects
            period (int):
        """
        self._writers = writers
        for w in writers:
            assert isinstance(w, EventWriter), w
        self._period = period

    def after_step(self):
        if (
            (self.trainer.iter + 1) % self._period == 0
            or (self.trainer.iter == self.trainer.max_iters - 1)
            or (self.trainer.iter == 0)
        ):
            for writer in self._writers:
                writer.write(self._period)

    def after_train(self):
        for writer in self._writers:
            writer.close()


@HOOKS.register()
class PeriodicCheckpoint(HookBase):
    """
    It is executed every ``period`` iterations and after the last iteration.
    """

    def __init__(self, period):
        self.period = int(period)

    def before_train(self):
        self.max_iters = self.trainer.max_iters
        self.max_epochs = self.trainer.max_epochs

    def after_step(self):
        # No way to use **kwargs
        """
        Perform the appropriate action at the given iteration.
        Args:
            iteration (int): the current iteration, ranged in [0, max_iter-1].
            kwargs (Any): extra data to save, same as in
                :meth:`Checkpointer.save`.
        """
        iteration = int(self.trainer.iter)
        additional_state = {"iteration": iteration}

        if (iteration + 1) % self.period == 0:
            ckpt_name = "model_{:07d}".format(iteration + 1)
            self.trainer.checkpointer.save(ckpt_name, **additional_state)

        if iteration >= self.max_iters - 1:
            self.trainer.checkpointer.save("model_final", **additional_state)


class EvalHook(HookBase):
    """
    Run an evaluation function periodically, and at the end of training.
    It is executed every ``eval_period`` iterations and after the last iteration.
    """

    def __init__(self, eval_period, eval_function):
        """
        Args:
            eval_period (int): the period to run `eval_function`.
            eval_function (callable): a function which takes no arguments, and
                returns a nested dict of evaluation metrics.
        Note:
            This hook must be enabled in all or none workers.
            If you would like only certain workers to perform evaluation,
            give other workers a no-op function (`eval_function=lambda: None`).
        """
        self._period = eval_period
        self._func = eval_function

    def _do_eval(self):
        results = self._func()

        if results:
            if isinstance(results, dict):
                assert isinstance(results, dict), "Eval function must return a dict. Got {} instead.".format(results)

                flattened_results = flatten_results_dict(results)
                for k, v in flattened_results.items():
                    try:
                        v = float(v)
                    except Exception:
                        raise ValueError(
                            "[EvalHook] eval_function should return a nested dict of float. "
                            "Got '{}: {}' instead.".format(k, v)
                        )
                self.trainer.storage.put_scalars(**flattened_results, smoothing_hint=False)

        # Evaluation may take different time among workers.
        # A barrier make them start the next iteration together.
        comm.synchronize()

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = (next_iter == self.trainer.max_iter) and (self._period >= 0)
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_eval()

    def after_train(self):
        # func is likely a closure that holds reference to the trainer
        # therefore we clean it to avoid circular reference in the end
        del self._func
