import torch

from efg.utils.distributed import reduce_dict


class MetricBase:
    """Base class to be inherited by all metrics registered to Pythia. See
    the description on top of the file for more information. Child class must
    implement ``calculate`` function.

    Args:
        name (str): Name of the metric.

    """

    def __init__(self, name, params={}):
        self.name = name
        for kk, vv in params.items():
            setattr(self, kk, vv)

    def calculate(self, output, target, *args, **kwargs):
        # Override in your child class
        raise NotImplementedError("'calculate' must be implemented in the child class")

    def __call__(self, *args, **kwargs):
        with torch.no_grad():
            metric = self.calculate(*args, **kwargs) / self.iter_per_update
            output = {self.name: metric}
            output = reduce_dict(output)
        return output


class Accuracy(MetricBase):
    def __init__(self, iter_per_update=1):
        defaults = dict(iter_per_update=iter_per_update)
        super().__init__("accuracy", defaults)

    def calculate(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        if target.numel() == 0:
            return torch.zeros([], device=output.device)
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res[0]
