from torch.utils.data.sampler import Sampler
from efg.utils import distributed as comm
from efg.data.registry import SAMPLERS


@SAMPLERS.register()
class SeqInferenceSampler(Sampler):
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

        if self._rank == 0:
            begin = 0
            end = 4931
        elif self._rank == 1:
            begin = 4931
            end = 10076
        elif self._rank == 2:
            begin = 10076
            end = 15019
        elif self._rank == 3:
            begin = 15019
            end = 19962
        elif self._rank == 4:
            begin = 19962
            end = 24917
        elif self._rank == 5:
            begin = 24917
            end = 29878
        elif self._rank == 6:
            begin = 29878
            end = 34833
        elif self._rank == 7:
            begin = 34833
            end = 39987
        self._local_indices = range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)
