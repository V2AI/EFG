import torch
import itertools
import os
import uuid
from pathlib import Path
import numpy as np
from tqdm import tqdm
from efg.evaluator.waymo_evaluator import WaymoDetEvaluator
from efg.evaluator.registry import EVALUATORS
from efg.data.datasets.waymo import CAT_TO_IDX, LABEL_TO_TYPE
from efg.geometry.box_ops_torch import limit_period
from efg.utils import distributed as comm


@EVALUATORS.register()
class CustomWaymoTrackEvaluator(WaymoDetEvaluator):
    def __init__(self, config, output_dir, dataset=None):
        self.config = config
        self._distributed = comm.is_dist_avail_and_initialized()
        self._output_dir = output_dir
        self.res_path = str(Path(self._output_dir) / Path("results.pth"))
        self._cpu_device = torch.device("cpu")
        self._classes = config.dataset.classes
        self._local_eval = True
        self.root_path = self.config.detection.source.local5f.root
        self.val_path = self.config.detection.source.local5f.val
        self.metrics_path = self.config.trainer.eval_metrics_path
        self.eval_class = self.config.model.eval_class

    def evaluate(self):
        if self._distributed:
            comm.synchronize()

            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            self._infos = comm.gather(self._infos, dst=0)
            self._infos = list(itertools.chain(*self._infos))

            if not comm.is_main_process():
                return {}

        if self._local_eval:
            classes = np.array([CAT_TO_IDX[name] for name in self._classes])

            for target, output in zip(self._infos, self._predictions):
                target["annotations"]["labels"] = np.array(
                    [
                        LABEL_TO_TYPE[label]
                        for label in target["annotations"]["labels"]
                    ]
                )

                target["annotations"]["gt_boxes"][:, -1] = limit_period(
                    target["annotations"]["gt_boxes"][:, -1],
                    offset=0.5,
                    period=np.pi * 2,
                )

                if self.eval_class == "VEHICLE":
                    mask = output["track_labels"] == 1
                elif self.eval_class == "PEDESTRIAN":
                    mask = output["track_labels"] == 2
                elif self.eval_class == "CYCLIST":
                    mask = output["track_labels"] == 3
                output["track_labels"] = output["track_labels"][mask]
                output["track_scores"] = output["track_scores"][mask]
                output["track_boxes3d"] = output["track_boxes3d"][mask]

            processed_results = {
                target["metadata"]["token"]: {
                    "track_scores": output["track_scores"],
                    "track_labels": output["track_labels"],
                    "track_boxes3d": output["track_boxes3d"],
                    "track_ids": output["track_ids"],
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
            print("Start local waymo tracking evaluation...")
            result_path = self.create_pd_detection(
                processed_results, self._output_dir, name="pred"
            )
            gt_path = self.create_gt_detection(
                processed_results, self._output_dir, name="gt"
            )
            cmd = "%s  %s  %s" % (self.metrics_path, result_path, gt_path)
            result = os.popen(cmd).read()
            print(result)

            with open(self._output_dir + "/tracking_result.txt", "w") as f:
                f.write(result)

    def create_pd_detection(self, detections, result_path, name, tracking=True):
        """Creates a prediction objects file."""
        from waymo_open_dataset import label_pb2
        from waymo_open_dataset.protos import metrics_pb2

        objects = metrics_pb2.Objects()
        infos = np.load(self.root_path + self.val_path, "rb", True)
        infos = reorganize_info(infos)
        for token, detection in tqdm(detections.items()):
            info = infos[token]
            box3d = detection["track_boxes3d"].numpy()
            scores = detection["track_scores"].numpy()
            labels = detection["track_labels"].numpy()

            if tracking:
                tracking_ids = detection["track_ids"]

            for i in range(box3d.shape[0]):
                det = box3d[i]
                score = scores[i]

                label = int(labels[i])

                o = metrics_pb2.Object()
                o.context_name = info["scene_name"]
                o.frame_timestamp_micros = info["frame_name"]

                # Populating box and score.
                box = label_pb2.Label.Box()
                box.center_x = det[0]
                box.center_y = det[1]
                box.center_z = det[2]
                box.length = det[3]
                box.width = det[4]
                box.height = det[5]
                box.heading = det[-1]
                o.object.box.CopyFrom(box)
                o.score = score
                # Use correct type.
                o.object.type = LABEL_TO_TYPE[label]

                if tracking:
                    o.object.id = uuid_gen.get_uuid(int(tracking_ids[i]))

                objects.objects.append(o)

        # Write objects to a file.
        path = os.path.join(result_path, "%s.bin" % name)
        print("PRED results saved to {}".format(path))
        f = open(path, "wb")
        f.write(objects.SerializeToString())
        f.close()
        return path

    def create_gt_detection(self, detections, result_path, name, tracking=True):
        """Creates a gt prediction object file for local evaluation."""
        from waymo_open_dataset import label_pb2
        from waymo_open_dataset.protos import metrics_pb2

        objects = metrics_pb2.Objects()
        infos = np.load(self.root_path + self.val_path, "rb", True)
        infos = reorganize_info(infos)
        for token, detection in tqdm(detections.items()):
            info = infos[token]
            names = info["annotations"]["gt_names"]
            num_points_in_gt = info["annotations"]["num_points_in_gt"]
            box3d = info["annotations"]["gt_boxes"][:, [0, 1, 2, 3, 4, 5, -1]]

            if len(box3d) == 0:
                continue

            for i in range(box3d.shape[0]):
                if num_points_in_gt[i] == 0:
                    continue
                if names[i] == "UNKNOWN":
                    continue

                det = box3d[i]
                score = 1.0
                label = names[i]

                o = metrics_pb2.Object()
                o.context_name = info["scene_name"]
                o.frame_timestamp_micros = info["frame_name"]

                # Populating box and score.
                box = label_pb2.Label.Box()
                box.center_x = det[0]
                box.center_y = det[1]
                box.center_z = det[2]
                box.length = det[3]
                box.width = det[4]
                box.height = det[5]
                box.heading = det[-1]
                o.object.box.CopyFrom(box)
                o.score = score
                # Use correct type.
                o.object.type = CAT_TO_IDX[label]
                o.object.num_lidar_points_in_box = num_points_in_gt[i]
                if tracking:
                    o.object.id = info["annotations"]["gt_ids"][i]

                objects.objects.append(o)

        # Write objects to a file.
        path = os.path.join(result_path, "%s.bin" % name)
        print("GT results saved to {}".format(path))
        f = open(path, "wb")
        f.write(objects.SerializeToString())
        f.close()
        return path


def reorganize_info(infos):
    new_info = {}

    for info in infos:
        token = info["token"]
        new_info[token] = info

    return new_info


class UUIDGeneration:
    def __init__(self):
        self.mapping = {}

    def get_uuid(self, seed):
        if seed not in self.mapping:
            self.mapping[seed] = uuid.uuid4().hex
        return self.mapping[seed]


uuid_gen = UUIDGeneration()
