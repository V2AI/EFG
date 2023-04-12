import math

import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler

from efg.utils import distributed as comm

from .registry import SAMPLERS


@SAMPLERS.register()
class InfiniteSampler(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        else:
            perm = torch.arange(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)

    next = __next__  # Python 2 compatibility


@SAMPLERS.register()
class DistributedInfiniteSampler(InfiniteSampler):
    def __init__(self, data_source, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.data_source = data_source
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.it = 0
        self.num_samples = int(math.ceil(len(self.data_source) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.reset_permutation()

    def __next__(self):
        it = self.it * self.num_replicas + self.rank
        value = self._perm[it % len(self._perm)]
        self.it = self.it + 1

        if (self.it * self.num_replicas) >= len(self._perm):
            self.reset_permutation()
            self.it = 0
        return value

    def __len__(self):
        return self.num_samples


@SAMPLERS.register()
class InferenceSampler(Sampler):
    """
    Produce indices for inference.
    Inference needs to run on the __exact__ set of samples,
    therefore when the total number of samples is not divisible by the number of workers,
    this sampler produces different number of samples on different workers.
    """

    def __init__(self, size: int):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
        """
        self._size = size
        assert size > 0
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        shard_size = (self._size - 1) // self._world_size + 1
        begin = shard_size * self._rank
        end = min(shard_size * (self._rank + 1), self._size)
        self._local_indices = range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


@SAMPLERS.register()
class DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    """

    def __init__(self, dataset, samples_per_gpu=1, num_replicas=None, rank=None):
        """
        Args:
            dataset (Dataset): Dataset used for sampling.
            num_replicas (optional): Number of processes participating in
                distributed training.
            rank (optional): Rank of the current process within num_replicas.
        """
        _rank = comm.get_rank()
        _num_replicas = comm.get_world_size()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        assert hasattr(self.dataset, "aspect_ratios")
        self.aspect_ratios = self.dataset.aspect_ratios
        self.group_sizes = np.bincount(self.aspect_ratios)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += (
                int(math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu / self.num_replicas))
                * self.samples_per_gpu
            )
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.aspect_ratios == i)[0]
                assert len(indice) == size
                indice = indice[list(torch.randperm(int(size), generator=g))].tolist()
                extra = int(
                    math.ceil(size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                # pad indice
                tmp = indice.copy()
                for _ in range(extra // size):
                    indice.extend(tmp)
                indice.extend(tmp[: extra % size])
                indices.extend(indice)

        assert len(indices) == self.total_size

        indices = [
            indices[j]
            for i in list(torch.randperm(len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) * self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
