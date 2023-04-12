import itertools
import logging
import os
from collections import abc
from pathlib import Path
from typing import Any, Dict

import numpy as np

import torch

from efg.data.datasets.waymo import CAT_TO_IDX, LABEL_TO_TYPE
from efg.geometry.box_ops_torch import limit_period
from efg.utils import distributed as comm

from .evaluator import DatasetEvaluator
from .registry import EVALUATORS

logger = logging.getLogger(__name__)


@EVALUATORS.register()
class WaymoDetEvaluator(DatasetEvaluator):
    def __init__(self, config, output_dir, dataset=None):
        self.config = config
        self._distributed = comm.is_dist_avail_and_initialized()
        self._output_dir = output_dir
        self.res_path = str(Path(self._output_dir) / Path("results.pth"))
        self._cpu_device = torch.device("cpu")
        self._classes = config.dataset.classes
        self._local_eval = config.task == "val"

    def reset(self):
        self._predictions = []
        self._infos = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            self._infos.append(input[1])
            self._predictions.append(output)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()

            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            self._infos = comm.gather(self._infos, dst=0)
            self._infos = list(itertools.chain(*self._infos))

            if not comm.is_main_process():
                return {}

        classes = np.array([CAT_TO_IDX[name] for name in self._classes])

        for target, output in zip(self._infos, self._predictions):
            # project 0, 1, 2 to 1, 2, 3
            output["labels"] = torch.tensor([LABEL_TO_TYPE[label] for label in output["labels"].numpy()])
            target["annotations"]["labels"] = np.array(
                [LABEL_TO_TYPE[label] for label in target["annotations"]["labels"]]
            )
            target["annotations"]["gt_boxes"][:, -1] = limit_period(
                target["annotations"]["gt_boxes"][:, -1], offset=0.5, period=np.pi * 2
            )

        processed_results = {
            target["metadata"]["token"]: {
                "pred_scores": output["scores"],
                "pred_labels": output["labels"],
                "pred_boxes3d": output["boxes3d"],
                "metadata": target["metadata"],
                "boxes3d": target["annotations"]["gt_boxes"],
                "labels": target["annotations"]["labels"],
                "difficulty": target["annotations"]["difficulty"],
                "num_points_in_gt": target["annotations"]["num_points_in_gt"],
                "classes": classes,
            }
            for target, output in zip(self._infos, self._predictions)
        }

        torch.save(processed_results, self.res_path)
        logger.info("Start local waymo evaluation...")
        eval_script = os.path.join(os.environ["EFG_PATH"], "datasets/utils/waymo_eval.py")
        cmd = "python " + eval_script + f" --root-path log/inference --seed {self.config.misc.seed}"
        os.system(cmd)


def recursive_copy_to_device(value, non_blocking, device):
    """
    Recursively searches lists, tuples, dicts and copies tensors to device if
    possible. Non-tensor values are passed as-is in the result.
    NOTE:  These are all copies, so if there are two objects that reference
    the same object, then after this call, there will be two different objects
    referenced on the device.
    """

    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)

    if isinstance(value, (list, tuple)):
        values = []
        for val in value:
            values.append(recursive_copy_to_device(val, non_blocking=non_blocking, device=device))

        return values if isinstance(value, list) else tuple(values)

    if isinstance(value, abc.Mapping):
        device_val: Dict[str, Any] = {}
        for key, val in value.items():
            device_val[key] = recursive_copy_to_device(val, non_blocking=non_blocking, device=device)

        return device_val

    return value
