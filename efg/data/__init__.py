import logging
import os
import random
from datetime import datetime

import numpy as np

import torch
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from efg.utils import distributed as comm

from .augmentations import *
from .augmentations3d import *
from .base_dataset import BaseDataset
from .datasets.coco import COCODataset
from .datasets.nuscenes.nuscenes import nuScenesDetectionDataset
from .datasets.waymo import WaymoDetectionDataset
from .registry import DATASETS, SAMPLERS
from .sampler import DistributedInfiniteSampler, InferenceSampler

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


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def worker_init_reset_seed(worker_id):
    seed_all_rng(torch.initial_seed() % 2**32)
    # seed_all_rng(np.random.randint(2**31) + worker_id)


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
    return DATASETS.get(config.dataset.type)(config)


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
