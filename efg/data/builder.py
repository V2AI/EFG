import logging
import os
import random
from datetime import datetime
from types import SimpleNamespace

import numpy as np
from omegaconf import DictConfig

import torch
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from efg.utils import distributed as comm

from .registry import DATASETS, PROCESSORS, SAMPLERS

logger = logging.getLogger(__name__)


@DATASETS.register()
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.
    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.
    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
    """

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(datasets)
        if hasattr(self.datasets[0], "aspect_ratios"):
            aspect_ratios = [d.aspect_ratios for d in self.datasets]
            self.aspect_ratios = np.concatenate(aspect_ratios)
        if hasattr(self.datasets[0], "meta"):
            meta = {}
            for d in self.datasets:
                meta.update(**d.meta.__dict__)
            self.meta = SimpleNamespace(**meta)


@DATASETS.register()
class RepeatDataset(object):
    """A wrapper of repeated dataset.
    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.
    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        if hasattr(self.dataset, "aspect_ratios"):
            self.aspect_ratios = np.tile(self.dataset.aspect_ratios, times)

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        return self.times * self._ori_len


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2**31) + worker_id)


def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.
    Args:
        seed (int): if None, will use a strong random seed.
    Returns:
        seed (int): used seed value.
    """
    if seed is None:
        seed = os.getpid() + int(datetime.now().strftime("%S%f")) + int.from_bytes(os.urandom(2), "big")
        logger.info(f"Using the random generated seed {seed}")
    else:
        logger.info(f"Using a preset seed {seed}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.set_rng_state(torch.manual_seed(seed).get_state())

    return seed


def build_dataset(config):
    if hasattr(config.dataset, "source"):  # single dataset back compatibility
        return DATASETS.get(config.dataset.type)(config)
    else:
        datasets = []
        assert hasattr(config.dataset, "sources")  # datasets list
        dataset_cfgs = [ds for ds in config.dataset.sources]
        for dscfg in dataset_cfgs:
            dataset_type = dscfg.pop("dataset")
            config.dataset.source = dscfg.pop("source")
            dataset = DATASETS.get(dataset_type)(config, **dscfg)
            datasets.append(dataset)
        return DATASETS.get(config.dataset.compose_type)(datasets)


def build_dataloader(config, dataset, msg=False):
    if config.task == "train":
        num_devices = comm.get_world_size()
        rank = comm.get_rank()
        sampler_name = config.dataloader.sampler
        if sampler_name == "DistributedGroupSampler":
            sampler = SAMPLERS.get(sampler_name)(
                dataset, config.dataloader.batch_size, comm.get_world_size(), comm.get_rank()
            )
        else:
            sampler = SAMPLERS.get(config.dataloader.sampler)(dataset, num_devices, rank)
    else:
        sampler = SAMPLERS.get(config.dataloader.eval_sampler)(len(dataset))
        config.dataloader.batch_size = 1

    if msg:
        g = torch.Generator()
        g.manual_seed(0)
    else:
        g = None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.dataloader.batch_size,
        sampler=sampler,
        num_workers=config.dataloader.num_workers if not config.misc.debug else 0,
        collate_fn=trivial_batch_collator,
        worker_init_fn=worker_init_reset_seed,
        generator=g,
        pin_memory=True,
    )

    return data_loader


def build_processors(pipelines):
    transforms = []
    for pipeline in pipelines:
        if isinstance(pipeline, (dict, DictConfig)):
            name, args = pipeline.copy().popitem()
            transform = PROCESSORS.get(name)(**args)
            transforms.append(transform)
        else:
            transform = PROCESSORS.get(pipeline)()
            transforms.append(transform)

    return transforms
