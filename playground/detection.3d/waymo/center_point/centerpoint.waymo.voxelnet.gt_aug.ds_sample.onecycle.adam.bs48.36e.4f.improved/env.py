from efg.data.datasets.waymo import WaymoDetectionDataset
from efg.data.registry import DATASETS
from efg.engine.registry import TRAINERS
from efg.engine.trainer import DefaultTrainer


@TRAINERS.register()
class CustomTrainer(DefaultTrainer):
    def __init__(self, configuration):
        super(CustomTrainer, self).__init__(configuration)
        if self.is_train:
            self.fade_start_iter = int(self.max_iters * (1 - self.config.trainer.fade))

    def step(self):
        if (
            self.iter > self.fade_start_iter
            and len(self.dataloader.dataset.transforms) == self.dataloader.dataset.transforms_length
        ):
            self.dataloader.dataset.transforms = self.dataloader.dataset.transforms[1:]
            self._dataiter = iter(self.dataloader)

        super().step()


@DATASETS.register()
class CustomWDDataset(WaymoDetectionDataset):
    def __init__(self, config):
        super(CustomWDDataset, self).__init__(config)
        self.transforms_length = len(self.transforms)
