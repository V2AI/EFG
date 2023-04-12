import itertools
import json
import logging
import operator
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box

import torch

from efg.evaluator.evaluator import DatasetEvaluator
from efg.evaluator.registry import EVALUATORS
from efg.utils import distributed as comm


def eval_main(nusc, eval_version, res_path, eval_set, output_dir):
    # nusc = NuScenes(version=version, dataroot=str(root_path), verbose=True)
    cfg = config_factory(eval_version)

    nusc_eval = NuScenesEval(
        nusc,
        config=cfg,
        result_path=res_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=True,
    )
    metrics_summary = nusc_eval.main(plot_examples=0)
    return metrics_summary


def _lidar_nusc_box_to_global(nusc, boxes, sample_token):
    try:
        s_record = nusc.get("sample", sample_token)
        sample_data_token = s_record["data"]["LIDAR_TOP"]
    except:
        sample_data_token = sample_token

    sd_record = nusc.get("sample_data", sample_data_token)
    cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(Quaternion(cs_record["rotation"]))
        box.translate(np.array(cs_record["translation"]))
        # Move box to global coord system
        box.rotate(Quaternion(pose_record["rotation"]))
        box.translate(np.array(pose_record["translation"]))
        box_list.append(box)
    return box_list


def _efg_det_to_nusc_box(detection):
    box3d = detection["box3d_lidar"].detach().cpu().numpy()
    scores = detection["scores"].detach().cpu().numpy()
    labels = detection["label_preds"].detach().cpu().numpy()

    box_list = []
    rot = Quaternion(axis=[0, 0, 1], degrees=90)
    for i in range(box3d.shape[0]):
        quat = Quaternion(axis=[0, 0, 1], radians=box3d[i, -1])
        velocity = (*box3d[i, 6:8], 0.0)
        box = Box(
            box3d[i, :3],
            box3d[i, [4, 3, 5]],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
        )
        box.rotate(rot)
        box_list.append(box)
    return box_list


@EVALUATORS.register()
class nuScenesDetEvaluator(DatasetEvaluator):
    def __init__(self, config, output_dir, dataset=None):
        self.dataset_name = "nuscenes_detection_val"
        self._distributed = comm.is_dist_avail_and_initialized()
        self._output_dir = output_dir
        self._class_names = config.dataset.classes
        self.res_path = str(Path(self._output_dir) / Path(self.dataset_name + ".json"))
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = dataset.meta
        self.eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
            "v1.0-test": "test",
        }

        self.nusc_annos = {
            "results": {},
            "meta": None,
        }

    def reset(self):
        self._predictions = []
        self._dump_infos = []  # per task

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            output["token"] = input[1]["token"]
            self._predictions.append(output)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()

            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return {}

        root_path = self._metadata["root_path"]

        nusc = NuScenes(version=self._metadata["version"], dataroot=root_path, verbose=True)

        for det in tqdm(self._predictions):
            annos = []
            boxes = _efg_det_to_nusc_box(det)
            boxes = _lidar_nusc_box_to_global(nusc, boxes, det["token"])
            for i, box in enumerate(boxes):
                name = self._metadata["mapped_class_names"][box.label]
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in ["car", "construction_vehicle", "bus", "truck", "trailer"]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = None
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = None

                nusc_anno = {
                    "sample_token": det["token"],
                    "translation": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "velocity": box.velocity[:2].tolist(),
                    "detection_name": name,
                    "detection_score": box.score,
                    "attribute_name": attr
                    if attr is not None
                    else max(self._metadata["cls_attr_dist"][name].items(), key=operator.itemgetter(1))[0],
                }
                annos.append(nusc_anno)
            self.nusc_annos["results"].update({det["token"]: annos})

        self.nusc_annos["meta"] = {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }

        with open(self.res_path, "w") as f:
            json.dump(self.nusc_annos, f)

        print("Start Evaluation...")
        metrics_summary = eval_main(
            nusc,
            self._metadata["evaluator_type"],
            self.res_path,
            self.eval_set_map[self._metadata["version"]],
            self._output_dir,
        )

        return metrics_summary
