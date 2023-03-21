import numpy as np


class HistoryBuffer:
    """
    Track a series of scalar values and provide access to smoothed values over a
    window or the global average of the series.
    """

    def __init__(self, max_length: int = 1000000):
        """
        Args:
            max_length: maximal number of values that can be stored in the
                buffer. When the capacity of the buffer is exhausted, old
                values will be removed.
        """
        self._max_length: int = max_length
        self._data = []  # (value, iteration) pairs
        self._count = 0
        self._global_avg = 0.0

    def update(self, value, iteration=None):
        """
        Add a new scalar value produced at certain iteration. If the length
        of the buffer exceeds self._max_length, the oldest element will be
        removed from the buffer.
        """
        if iteration is None:
            iteration = self._count
        if len(self._data) == self._max_length:
            self._data.pop(0)
        self._data.append((value, iteration))

        self._count += 1
        if not isinstance(value, list):
            self._global_avg += (value - self._global_avg) / self._count
        else:
            if isinstance(self._global_avg, float):
                self._global_avg = [self._global_avg] * len(value)
            self._global_avg = [avg + (v - avg) / self._count for v, avg in zip(value, self._global_avg)]

    def latest(self):
        """
        Return the latest scalar value added to the buffer.
        """
        return self._data[-1][0]

    def median(self, window_size: int):
        """
        Return the median of the latest `window_size` values in the buffer.
        """
        return np.median([x[0] for x in self._data[-window_size:]])

    def avg(self, window_size: int):
        """
        Return the mean of the latest `window_size` values in the buffer.
        """
        return np.mean([x[0] for x in self._data[-window_size:]])

    def global_avg(self):
        """
        Return the mean of all the elements in the buffer. Note that this
        includes those getting removed due to limited buffer storage.
        """
        return self._global_avg

    def values(self):
        """
        Returns:
            list[(number, iteration)]: content of the current buffer.
        """
        return self._data
