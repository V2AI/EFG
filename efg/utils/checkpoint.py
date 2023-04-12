import copy
import logging
import os
import pickle

import numpy as np
from omegaconf import OmegaConf

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from efg.utils import distributed as comm
from efg.utils.file_io import PathManager

from .checkpoint_utils import (
    _strip_prefix_if_present,
    get_missing_parameters_message,
    get_unexpected_parameters_message
)
from .d2_model_loading import align_and_update_state_dicts

logger = logging.getLogger(__name__)


class Checkpointer:
    def __init__(self, model, save_dir, **checkpointables):
        """
        Generates a path for saving model which can also be used for resuming
        from a checkpoint.
        """
        if isinstance(model, (DistributedDataParallel, DataParallel)):
            model = model.module
        self.model = model
        self.save_dir = save_dir
        self.checkpointables = copy.copy(checkpointables)

    def save_config(self):
        if not comm.get_rank() == 0:
            return

        cfg_file = os.path.join(self.ckpt_foldername, "config.yaml")
        with PathManager.open(cfg_file, "w") as f:
            f.write(OmegaConf.to_yaml(self.config, resolve=True))

    def save(self, file_name, **kwargs):
        data = {}
        data["model"] = self.model.state_dict()
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        file_name = f"{file_name}.pth"
        save_file = os.path.join(self.save_dir, file_name)
        logger.info("Saving checkpoint to {}".format(save_file))
        with PathManager.open(save_file, "wb") as f:
            torch.save(data, f)

    def load(self, file_path: str, resume=False):
        if not file_path:
            logger.info("No checkpoint found. Initializing model from scratch.")
            return {}
        logger.info(f"Loading checkpoint from {file_path}")

        if not os.path.isfile(file_path):
            file_path = PathManager.get_local_path(file_path)
            assert PathManager.isfile(file_path), f"Checkpoint {file_path} not found!"

        checkpoint = self._load_file(file_path)

        if checkpoint.get("matching_heuristics", False):
            _convert_ndarray_to_tensor(checkpoint["model"])
            # convert weights by name-matching heuristics
            model_state_dict = self.model.state_dict()
            align_and_update_state_dicts(
                model_state_dict,
                checkpoint["model"],
                c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
            )
            checkpoint["model"] = model_state_dict

        self._load_model(checkpoint)

        if resume:
            for key, obj in self.checkpointables.items():
                if key in checkpoint:
                    logger.info(f"Loading {key} from {file_path}")
                    obj.load_state_dict(checkpoint.pop(key))
            # return any further checkpoint data
            return checkpoint
        else:
            return {}

    def _load_file(self, filename):
        """
        Args:
            filename (str): load checkpoint file from local or oss. checkpoint can be of type
                pkl, pth
        """
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in cvpods model zoo format
                logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}
        elif filename.endswith(".pth"):
            if filename.startswith("s3://"):
                with PathManager.open(filename, "rb") as f:
                    loaded = torch.load(f, map_location=torch.device("cpu"))
            else:
                loaded = torch.load(filename, map_location=torch.device("cpu"))
            if "model" not in loaded:
                loaded = {"model": loaded}
            return loaded

    def _load_model(self, checkpoint):
        """
        Load weights from a checkpoint.
        Args:
            checkpoint: checkpoint contains the weights.
        """
        checkpoint_state_dict = checkpoint.pop("model")
        _convert_ndarray_to_tensor(checkpoint_state_dict)

        # if the state_dict comes from a model that was wrapped in a
        # DataParallel or DistributedDataParallel during serialization,
        # remove the "module" prefix before performing the matching.
        _strip_prefix_if_present(checkpoint_state_dict, "module.")
        model_state_dict = self.model.state_dict()

        # work around https://github.com/pytorch/pytorch/issues/24139
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    logger.warning(
                        "'{}' has shape {} in the checkpoint but {} in the "
                        "model! Skipped.".format(k, shape_checkpoint, shape_model)
                    )
                    checkpoint_state_dict.pop(k)

        incompatible = self.model.load_state_dict(checkpoint_state_dict, strict=False)

        if incompatible.missing_keys:
            logger.info(get_missing_parameters_message(incompatible.missing_keys))
        if incompatible.unexpected_keys:
            logger.info(get_unexpected_parameters_message(incompatible.unexpected_keys))


def _convert_ndarray_to_tensor(state_dict: dict):
    """
    In-place convert all numpy arrays in the state_dict to torch tensor.
    Args:
        state_dict (dict): a state-dict to be loaded to the model.
    """
    # model could be an OrderedDict with _metadata attribute
    # (as returned by Pytorch's state_dict()). We should preserve these
    # properties.
    for k in list(state_dict.keys()):
        if "weight_order" in k:
            continue
        v = state_dict[k]
        if not isinstance(v, np.ndarray) and not isinstance(v, torch.Tensor):
            raise ValueError("Unsupported type found in checkpoint! {}: {}".format(k, type(v)))
        if not isinstance(v, torch.Tensor):
            state_dict[k] = torch.from_numpy(v)
