import logging
import os
import socket
import time
import weakref
from typing import Dict

import numpy as np

import torch
from torch.nn.parallel import DistributedDataParallel

from efg.data import build_dataloader, build_dataset
from efg.engine.hooks import HookBase, IterTimer, LRScheduler, Optimization, PeriodicCheckpoint, PeriodicWriter
from efg.evaluator.evaluator import inference_on_dataset
from efg.solver import build_optimizer, build_scheduler
from efg.utils import distributed as comm
from efg.utils.checkpoint import Checkpointer
from efg.utils.events import CommonMetricPrinter, EventStorage, JSONWriter, TensorboardXWriter, get_event_storage
from efg.utils.file_io import PathManager

from .registry import TRAINERS

logger = logging.getLogger(__name__)


class TrainerBase:
    def __init__(self):
        self._hooks = []
        self._metrics = {}

    def register_hooks(self, hooks):
        """
        Register hooks to the runner. The hooks are executed in the order
        they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and runner cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self):
        """
        Args:
            resume: resume to train or not
        """
        self.model.train()
        logger.info(f"Start training from iter {self.start_iter}")

        with EventStorage(self.start_iter) as self.storage:
            self.before_train()
            for self.iter in range(self.start_iter, self.max_iters):
                if self.iter % (self.config.trainer.window_size * 20) == 0:
                    logger.info(f"Host Name: {socket.gethostname()}")
                    logger.info(f"Experiment Dir: {self.config.trainer.output_dir.split('EFG.private/')[-1]}")
                self.before_step()
                # by default, a step contains data_loading and model forward,
                # loss backward is executed in after_step for better expansibility
                self.step()
                self.after_step()
            # self.iter == max_iters can be used by `after_train` to
            # tell whether the training successfully finished or failed
            # due to exceptions.
            self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        self.storage._iter = self.iter
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        # Maintain the invariant that storage.iter == runner.iter
        # for the entire execution of each step
        self.storage._iter = self.iter

        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()

    def step(self):
        raise NotImplementedError

    def _write_metrics(self, loss_dict: Dict[str, torch.Tensor], time_dict: Dict[str, float], prefix: str = ""):
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
        """
        device = next(iter(loss_dict.values())).device

        # Use a new stream so these ops don't wait for DDP or backward
        with torch.cuda.stream(torch.cuda.Stream() if device.type == "cuda" else None):
            metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
            metrics_dict.update(time_dict)

            # Gather metrics among all workers for logging
            # This assumes we do DDP-style training, which is currently the only supported method in efg.
            all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            for k in time_dict.keys():
                v = np.max([x.pop(k) for x in all_metrics_dict])
                storage.put_scalar(k, v)

            # average the rest metrics
            metrics_dict = {k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()}
            total_losses_reduced = sum(loss for key, loss in metrics_dict.items() if "loss" in key)
            storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)


@TRAINERS.register()
class DefaultTrainer(TrainerBase):
    def __init__(self, config):
        super(DefaultTrainer, self).__init__()
        self.config = config
        self.is_train = self.config.task == "train"
        self.setup()
        logger.info("Finish trainer setup")

    def setup(self):
        self.setup_data()
        self.setup_model()
        self.setup_checkpointer()
        if self.is_train:
            if self.config.model.weights is not None:
                self.checkpointer.load(self.config.model.weights)
            self.iter = 0
            self.start_iter = 0
            self.setup_hooks()

    def setup_data(self, task=None):
        self.dataset = build_dataset(self.config)
        logger.info(f"Finish dataset setup: {self.dataset}")
        self.dataloader = build_dataloader(self.config, self.dataset)

        if self.is_train:
            max_epochs = self.config.solver.lr_scheduler.max_epochs
            if max_epochs is not None:
                max_iters = len(self.dataloader) * max_epochs
                self.config.solver.lr_scheduler.max_iters = max_iters
                self.config.solver.lr_scheduler.epoch_iters = len(self.dataloader)
                logger.info(f"Convert {max_epochs} epochs into {max_iters} iters")

            self._dataiter = iter(self.dataloader)
            logger.info(f"Finish dataloader setup: {self.dataloader}")

    def setup_model(self):
        model = self.build_model(self.config)

        if hasattr(self.dataset, "meta"):
            model.dataset_meta = self.dataset.meta

        if self.config.trainer.sync_bn and comm.is_dist_avail_and_initialized():
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            logger.info(f"Model is converted into syncbn version: {self.model}")
        else:
            self.model = model
            logger.info(f"Finish model setup: {self.model}")

        if self.is_train:
            self.optimizer = build_optimizer(self.config, self.model)
            logger.info(f"Finish optimizer setup: {type(self.optimizer)}")
            self.lr_scheduler = build_scheduler(self.config, self.optimizer)
            logger.info(f"Finish lr_scheduler setup: {self.lr_scheduler}")

        else:
            self.optimizer = None
            self.lr_scheduler = None

        if comm.get_world_size() > 1:
            torch.cuda.set_device(comm.get_local_rank())
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[comm.get_local_rank()],
                broadcast_buffers=False,
                find_unused_parameters=self.config.ddp.find_unused_parameters,
            )
            logger.info("Finish DDP setup")

    def setup_checkpointer(self):
        self.max_iters = self.config.solver.lr_scheduler.max_iters
        self.max_epochs = self.config.solver.lr_scheduler.get("max_epochs", None)
        self.checkpointer = Checkpointer(
            self.model,
            self.config.trainer.output_dir,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
        )

    def resume_or_load(self, resume=False):
        all_model_checkpoints = [
            os.path.join(self.config.trainer.output_dir, file)
            for file in PathManager.ls(self.config.trainer.output_dir)
            if PathManager.isfile(os.path.join(self.config.trainer.output_dir, file))
            and file.startswith("model_")
            and file.endswith(".pth")
        ]
        all_model_checkpoints = sorted(all_model_checkpoints, key=os.path.getmtime)

        if len(all_model_checkpoints) > 0:
            if self.config.model.weights is not None:
                matched = np.nonzero(
                    np.array([pts.endswith(self.config.model.weights.split("/")[-1]) for pts in all_model_checkpoints])
                )[0]
                if matched.shape[0] > 0:
                    load_path = all_model_checkpoints[matched[0]]
                else:
                    logger.info(f"Cannot find matched checkpoints as {self.config.model.weights}, load the latest.")
                    load_path = all_model_checkpoints[-1]
            else:
                load_path = all_model_checkpoints[-1]

            if resume and PathManager.isfile(load_path):
                self.start_iter = self.checkpointer.load(load_path, resume=resume).get("iteration", -1) + 1
                self.iter = self.start_iter
                if self.max_epochs is not None:
                    self.start_epochs = self.start_iter // len(self.dataloader)
            elif PathManager.isfile(load_path):
                self.checkpointer.load(load_path)
        elif PathManager.isfile(self.config.model.weights):
            self.checkpointer.load(self.config.model.weights)
        else:
            logger.info("Checkpoint does not exist")
            raise ModuleNotFoundError

    def setup_hooks(self):
        hooks = [
            Optimization(grad_clipper=self.config.solver.grad_clipper),
            LRScheduler(self.optimizer, self.lr_scheduler),
            IterTimer(),
        ]
        if comm.is_main_process():
            # setup periodic checkpointer
            if self.config.trainer.checkpoint_epoch is not None:
                checkpoint_period = self.config.trainer.checkpoint_epoch * len(self.dataloader)
            elif self.config.trainer.checkpoint_iter is not None:
                checkpoint_period = self.config.trainer.checkpoint_iter
            hooks.append(PeriodicCheckpoint(checkpoint_period))

            # run writers in the end, so that evaluation metrics are written
            window_size = self.config.trainer.window_size
            writers = [
                CommonMetricPrinter(self.max_iters),
                JSONWriter(os.path.join(self.config.trainer.output_dir, "metrics.json")),
                TensorboardXWriter(self.config.trainer.output_dir),
            ]
            hooks.append(PeriodicWriter(writers, period=window_size))

        # def test_and_save_results():
        #     self._last_eval_results = self.test(self.cfg, self.model)
        #     return self._last_eval_results
        # hooks.EvalHook(self.config.trainer.eval_period, test_and_save_results)

        self.register_hooks(hooks)
        logger.info(f"Finish hooks setup: {hooks}")

    def step(self):
        start = time.perf_counter()

        try:
            data = next(self._dataiter)
        except StopIteration:
            if hasattr(self.dataloader.sampler, "epoch"):
                self.dataloader.sampler.epoch += 1
            self._dataiter = iter(self.dataloader)
            data = next(self._dataiter)

        data_time = time.perf_counter() - start
        misc_dict = {
            "data_time": data_time,
        }

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        loss_dict = dict(sorted(loss_dict.items()))
        losses = sum([metrics_value for metrics_value in loss_dict.values() if metrics_value.requires_grad])
        self.outputs = {
            "losses": losses,
        }

        self._detect_anomaly(losses, loss_dict)
        self._write_metrics(loss_dict, misc_dict)

    def _detect_anomaly(self, losses, loss_dict):
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                "Loss became infinite or NaN at iteration={}!\nloss_dict = {}".format(self.iter, loss_dict)
            )

    def evaluate(self, evaluators, dataloader=None, test=False):
        model_without_ddp = self.model.module if comm.is_dist_avail_and_initialized() else self.model
        if hasattr(model_without_ddp, "is_train"):
            model_without_ddp.is_train = False
        inference_on_dataset(self.model, self.dataloader if dataloader is None else dataloader, evaluators)
