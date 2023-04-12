import contextlib
import copy
import datetime
import io
import itertools
import json
import logging
import os
import types

import numpy as np
from PIL import Image
from tabulate import tabulate
from termcolor import colored

import pycocotools.mask as mask_util

import torch

from efg.data.augmentations import build_processors
from efg.data.base_dataset import BaseDataset
from efg.data.datasets.builtin_meta import _get_builtin_metadata
from efg.data.detection_utils import annotations_to_instances, check_image_size, filter_empty_instances, read_image
from efg.data.registry import DATASETS
from efg.data.structures.boxes import Boxes, BoxMode
from efg.data.structures.masks import PolygonMasks
from efg.utils.file_io import PathManager, file_lock
from efg.utils.logger import log_first_n
from efg.utils.timer import Timer

"""
This file contains functions to parse COCO-format annotations into dicts in "efg format".
"""

logger = logging.getLogger(__name__)


@DATASETS.register()
class COCODataset(BaseDataset):
    def __init__(self, config, *args, **kwargs):
        super(COCODataset, self).__init__(config)

        self.is_train = config.task == "train"

        self.dataset_name = kwargs.get("dataset_name", "coco_2017") + ("_train" if self.is_train else "_val")
        if "panoptic" in self.dataset_name:
            self.task_key = "panoptic"  # for task: panoptic/semantic segmentation
        elif "keypoints" in self.dataset_name:
            self.task_key = "coco_person"  # for task: keypoints detection
        else:
            self.task_key = "coco"  # for task: instance detection/segmentation

        self.data_format = kwargs.get("format", "BGR")
        self.with_gt = kwargs.get("with_gt", True)
        self.use_instance_mask = kwargs.get("mask_on", False)
        self.instance_mask_format = kwargs.get("mask_format", "polygon")
        self.recompute_boxes = kwargs.get("recompute_boxes", False)

        if self.with_gt:
            if "panoptic" in self.dataset_name:
                metadata = _get_builtin_metadata("coco_panoptic_standard")
                data_root = config.dataset.source["root"]
                image_root, panoptic_root, json_file = config.dataset.source[config.task]
                metadata["image_root"] = data_root + image_root
                metadata["json_file"] = data_root + json_file
                metadata["panoptic_root"] = data_root + panoptic_root
                metadata["ignore_label"] = kwargs.get("ignore_label", 255)
                metadata["label_divisor"] = kwargs.get("label_divisor", 1000)
                meta = types.SimpleNamespace(**metadata)
                dataset_dicts = self._load_coco_panoptic_json(meta.json_file, meta.image_root, meta.panoptic_root, meta)
            elif self.task_key in ["coco", "coco_person"]:
                metadata = _get_builtin_metadata(self.task_key)
                data_root = config.dataset.source["root"]
                image_root, json_file = config.dataset.source[config.task]
                metadata["image_root"] = data_root + image_root
                metadata["json_file"] = data_root + json_file
                meta = types.SimpleNamespace(**metadata)
                dataset_dicts = self._load_coco_json(meta.json_file, meta.image_root, self.dataset_name, meta)

            self.meta = meta

            if self.is_train:
                self.dataset_dicts = self._filter_annotations(
                    dataset_dicts,
                    kwargs.get("filter_empty_annotations", True),
                )
            else:
                self.dataset_dicts = dataset_dicts
            self._set_group_flag()

        # build transforms
        self.transforms = build_processors(config.dataset.processors[config.task])
        logger.info(f"Building data processors: {self.transforms}")

    def _filter_annotations(self, dataset_dicts, filter_empty=True):
        """
        Load and prepare dataset dicts for instance detection/segmentation and
        semantic segmentation.

        Args:
            dataset_names (list[str]): a list of dataset names
            filter_empty (bool): whether to filter out images without instance annotations
        """
        has_instances = "annotations" in dataset_dicts[0]

        # Keep images without instance-level GT if the dataset has semantic labels
        # unless the task is panoptic segmentation.
        if filter_empty and has_instances:
            dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)

        if has_instances:
            class_names = self.meta.thing_classes
            # check_metadata_consistency("thing_classes", "coco", self.meta)
            print_instances_class_histogram(dataset_dicts, class_names)

        return dataset_dicts

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.aspect_ratios = np.zeros(len(self), dtype=np.uint8)
        if "width" in self.dataset_dicts[0] and "height" in self.dataset_dicts[0]:
            for i in range(len(self)):
                dataset_dict = self.dataset_dicts[i]
                if dataset_dict["width"] / dataset_dict["height"] > 1:
                    self.aspect_ratios[i] = 1

    def __getitem__(self, index):
        """Load data, apply transforms, converto to Instances."""
        dataset_dict = copy.deepcopy(self.dataset_dicts[index])

        # read image
        image = read_image(dataset_dict["file_name"], format=self.data_format)

        check_image_size(dataset_dict, image)

        if self.with_gt:
            if "annotations" in dataset_dict:
                annotations = dataset_dict.pop("annotations")
                annotations = [ann for ann in annotations if ann.get("iscrowd", 0) == 0]
            else:
                annotations = None

            if "sem_seg_file_name" in dataset_dict:
                if annotations is None:
                    annotations = []
                with PathManager.open(dataset_dict.get("sem_seg_file_name"), "rb") as f:
                    sem_seg_gt = Image.open(f)
                    sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
                annotations.insert(0, {"sem_seg": sem_seg_gt})
        else:
            annotations = None

        # apply transfrom
        image, annotations = self._apply_transforms(image, annotations, img_id=dataset_dict["image_id"])

        if self.with_gt:
            if "sem_seg_file_name" in dataset_dict:
                dataset_dict.pop("sem_seg_file_name")
                sem_seg_gt = annotations[0].pop("sem_seg")
                sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
                dataset_dict["sem_seg"] = sem_seg_gt
                annotations = annotations[1:]
                if len(annotations) == 0:
                    annotations = None

        if self.with_gt:
            # convert to Instance type
            if annotations is not None and len(annotations) > 0:  # got instances in annotations
                image_shape = image.shape[:2]  # h, w
                instances = annotations_to_instances(annotations, image_shape, mask_format=self.instance_mask_format)
                # # Create a tight bounding box from masks, useful when image is cropped
                # if self.crop_gen and instances.has("gt_masks"):
                #     instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                dataset_dict["instances"] = filter_empty_instances(instances)

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        if isinstance(image, list):
            image = np.stack(image)

        if image.shape[0] == 3:  # CHW
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image))
        elif len(image.shape) == 3 and image.shape[-1] == 3:
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        elif len(image.shape) == 4:
            if image.shape[-1] == 3:
                # NHWC -> NCHW
                dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(0, 3, 1, 2)))
            elif image.shape[1] == 3:
                # NCHW
                dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image))

        if not self.with_gt:
            dataset_dict["annotations"] = annotations

        return dataset_dict

    def __reset__(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset_dicts)

    def _load_coco_json(self, json_file, image_root, dataset_name=None, meta=None, extra_annotation_keys=None):
        """
        Load a json file with COCO's instances annotation format.
        Currently supports instance detection, instance segmentation,
        and person keypoints annotations.
        Args:
            json_file (str): full path to the json file in COCO instances annotation format.
            image_root (str or path-like): the directory where the images in this json file exists.
            dataset_name (str or None): the name of the dataset (e.g., coco_2017_train).
                When provided, this function will also do the following:
                * Put "thing_classes" into the metadata associated with this dataset.
                * Map the category ids into a contiguous range (needed by standard dataset format),
                  and add "thing_dataset_id_to_contiguous_id" to the metadata associated
                  with this dataset.
                This option should usually be provided, unless users need to load
                the original json content and apply more processing manually.
            extra_annotation_keys (list[str]): list of per-annotation keys that should also be
                loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
                "category_id", "segmentation"). The values for these keys will be returned as-is.
                For example, the densepose annotations are loaded in this way.
        Returns:
            list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
            `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
            If `dataset_name` is None, the returned `category_ids` may be
            incontiguous and may not conform to the Detectron2 standard format.
        Notes:
            1. This function does not read the image files.
               The results do not have the "image" field.
        """
        from pycocotools.coco import COCO

        timer = Timer()
        json_file = PathManager.get_local_path(json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO(json_file)
        if timer.seconds() > 1:
            logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

        id_map = None
        if dataset_name is not None:
            cat_ids = sorted(coco_api.getCatIds())
            cats = coco_api.loadCats(cat_ids)
            # The categories in a custom json file may not be sorted.
            thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
            meta.thing_classes = thing_classes

            # In COCO, certain category ids are artificially removed,
            # and by convention they are always ignored.
            # We deal with COCO's id issue and translate
            # the category ids to contiguous ids in [0, 80).

            # It works by looking at the "categories" field in the json, therefore
            # if users' own json also have incontiguous ids, we'll
            # apply this mapping as well but print a warning.
            if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
                if "coco" not in dataset_name:
                    logger.warning(
                        "Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you."
                    )
            id_map = {v: i for i, v in enumerate(cat_ids)}
            meta.thing_dataset_id_to_contiguous_id = id_map

        # sort indices for reproducible results
        img_ids = sorted(coco_api.imgs.keys())
        # imgs is a list of dicts, each looks something like:
        # {'license': 4,
        #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
        #  'file_name': 'COCO_val2014_000000001268.jpg',
        #  'height': 427,
        #  'width': 640,
        #  'date_captured': '2013-11-17 05:57:24',
        #  'id': 1268}
        imgs = coco_api.loadImgs(img_ids)
        # anns is a list[list[dict]], where each dict is an annotation
        # record for an object. The inner list enumerates the objects in an image
        # and the outer list enumerates over images. Example of anns[0]:
        # [{'segmentation': [[192.81,
        #     247.09,
        #     ...
        #     219.03,
        #     249.06]],
        #   'area': 1035.749,
        #   'iscrowd': 0,
        #   'image_id': 1268,
        #   'bbox': [192.81, 224.8, 74.73, 33.43],
        #   'category_id': 16,
        #   'id': 42986},
        #  ...]
        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
        total_num_valid_anns = sum([len(x) for x in anns])
        total_num_anns = len(coco_api.anns)
        if total_num_valid_anns < total_num_anns:
            logger.warning(
                f"{json_file} contains {total_num_anns} annotations, but only "
                f"{total_num_valid_anns} of them match to images in the file."
            )

        if "minival" not in json_file:
            # The popular valminusminival & minival annotations for COCO2014 contain this bug.
            # However the ratio of buggy annotations there is tiny and does not affect accuracy.
            # Therefore we explicitly white-list them.
            ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
            assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(json_file)

        imgs_anns = list(zip(imgs, anns))
        logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

        dataset_dicts = []

        ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])

        num_instances_without_valid_segmentation = 0

        for img_dict, anno_dict_list in imgs_anns:
            record = {}
            record["file_name"] = os.path.join(image_root, img_dict["file_name"])
            record["height"] = img_dict["height"]
            record["width"] = img_dict["width"]
            image_id = record["image_id"] = img_dict["id"]

            objs = []
            for anno in anno_dict_list:
                # Check that the image_id in this annotation is the same as
                # the image_id we're looking at.
                # This fails only when the data parsing logic or the annotation file is buggy.

                # The original COCO valminusminival2014 & minival2014 annotation files
                # actually contains bugs that, together with certain ways of using COCO API,
                # can trigger this assertion.
                assert anno["image_id"] == image_id

                assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

                obj = {key: anno[key] for key in ann_keys if key in anno}
                if "bbox" in obj and len(obj["bbox"]) == 0:
                    raise ValueError(
                        f"One annotation of image {image_id} contains empty 'bbox' value! "
                        "This json does not have valid COCO format."
                    )

                segm = anno.get("segmentation", None)
                if segm:  # either list[list[float]] or dict(RLE)
                    if isinstance(segm, dict):
                        if isinstance(segm["counts"], list):
                            # convert to compressed RLE
                            segm = mask_util.frPyObjects(segm, *segm["size"])
                    else:
                        # filter out invalid polygons (< 3 points)
                        segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                        if len(segm) == 0:
                            num_instances_without_valid_segmentation += 1
                            continue  # ignore this instance
                    obj["segmentation"] = segm

                keypts = anno.get("keypoints", None)
                if keypts:  # list[int]
                    for idx, v in enumerate(keypts):
                        if idx % 3 != 2:
                            # COCO's segmentation coordinates are floating points in [0, H or W],
                            # but keypoint coordinates are integers in [0, H-1 or W-1]
                            # Therefore we assume the coordinates are "pixel indices" and
                            # add 0.5 to convert to floating point coordinates.
                            keypts[idx] = v + 0.5
                    obj["keypoints"] = keypts

                obj["bbox_mode"] = BoxMode.XYWH_ABS
                if id_map:
                    annotation_category_id = obj["category_id"]
                    try:
                        obj["category_id"] = id_map[annotation_category_id]
                    except KeyError as e:
                        raise KeyError(
                            f"Encountered category_id={annotation_category_id} "
                            "but this id does not exist in 'categories' of the json file."
                        ) from e
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)

        if num_instances_without_valid_segmentation > 0:
            logger.warning(
                "Filtered out {} instances without valid segmentation. ".format(
                    num_instances_without_valid_segmentation
                )
                + "There might be issues in your dataset generation process.  Please "
                "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
            )
        return dataset_dicts

    @staticmethod
    def _load_coco_panoptic_json(json_file, image_dir, gt_dir, meta):
        """
        Args:
            image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
            gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
            json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
        Returns:
            list[dict]: a list of dicts in Detectron2 standard format. (See
            `Using Custom Datasets </tutorials/datasets.html>`_ )
        """

        def _convert_category_id(segment_info, meta):
            if segment_info["category_id"] in meta.thing_dataset_id_to_contiguous_id:
                segment_info["category_id"] = meta.thing_dataset_id_to_contiguous_id[segment_info["category_id"]]
                segment_info["isthing"] = True
            else:
                segment_info["category_id"] = meta.stuff_dataset_id_to_contiguous_id[segment_info["category_id"]]
                segment_info["isthing"] = False
            return segment_info

        with PathManager.open(json_file) as f:
            json_info = json.load(f)

        ret = []
        for ann in json_info["annotations"]:
            image_id = int(ann["image_id"])
            # TODO: currently we assume image and label has the same filename but
            # different extension, and images have extension ".jpg" for COCO. Need
            # to make image extension a user-provided argument if we extend this
            # function to support other COCO-like datasets.
            image_file = os.path.join(image_dir, os.path.splitext(ann["file_name"])[0] + ".jpg")
            label_file = os.path.join(gt_dir, ann["file_name"])
            segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
            ret.append(
                {
                    "file_name": image_file,
                    "image_id": image_id,
                    "pan_seg_file_name": label_file,
                    "segments_info": segments_info,
                }
            )
        assert len(ret), f"No images found in {image_dir}!"
        assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
        assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
        return ret


def convert_to_coco_dict(dataset_name, dataset_dicts, metadata):
    """
    Convert a dataset in efg's standard format into COCO json format
    COCO data format description can be found here:
    http://cocodataset.org/#format-data
    Args:
        dataset_name:
            name of the source dataset
            must be registered in DatastCatalog and in efg's standard format
    Returns:
        coco_dict: serializable dict in COCO json format
    """

    # unmap the category mapping ids for COCO
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
        reverse_id_mapping = {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}
        reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id]  # noqa
    else:
        reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa

    categories = [{"id": reverse_id_mapper(id), "name": name} for id, name in enumerate(metadata.thing_classes)]

    logger.info("Converting dataset dicts into COCO format")
    coco_images = []
    coco_annotations = []

    for image_id, image_dict in enumerate(dataset_dicts):
        coco_image = {
            "id": image_dict.get("image_id", image_id),
            "width": image_dict["width"],
            "height": image_dict["height"],
            "file_name": image_dict["file_name"],
        }
        coco_images.append(coco_image)

        anns_per_image = image_dict["annotations"]
        for annotation in anns_per_image:
            # create a new dict with only COCO fields
            coco_annotation = {}

            # COCO requirement: XYWH box format
            bbox = annotation["bbox"]
            bbox_mode = annotation["bbox_mode"]
            bbox = BoxMode.convert(bbox, bbox_mode, BoxMode.XYWH_ABS)

            # COCO requirement: instance area
            if "segmentation" in annotation:
                # Computing areas for instances by counting the pixels
                segmentation = annotation["segmentation"]
                # TODO: check segmentation type: RLE, BinaryMask or Polygon
                polygons = PolygonMasks([segmentation])
                area = polygons.area()[0].item()
            else:
                # Computing areas using bounding boxes
                bbox_xy = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                area = Boxes([bbox_xy]).area()[0].item()

            if "keypoints" in annotation:
                keypoints = annotation["keypoints"]  # list[int]
                for idx, v in enumerate(keypoints):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # For COCO format consistency we substract 0.5
                        # https://github.com/facebookresearch/detectron2/pull/175#issuecomment-551202163
                        keypoints[idx] = v - 0.5
                if "num_keypoints" in annotation:
                    num_keypoints = annotation["num_keypoints"]
                else:
                    num_keypoints = sum(kp > 0 for kp in keypoints[2::3])

            # COCO requirement:
            #   linking annotations to images
            #   "id" field must start with 1
            coco_annotation["id"] = len(coco_annotations) + 1
            coco_annotation["image_id"] = coco_image["id"]
            coco_annotation["bbox"] = [round(float(x), 3) for x in bbox]
            coco_annotation["area"] = area
            coco_annotation["category_id"] = reverse_id_mapper(annotation["category_id"])
            coco_annotation["iscrowd"] = annotation.get("iscrowd", 0)

            # Add optional fields
            if "keypoints" in annotation:
                coco_annotation["keypoints"] = keypoints
                coco_annotation["num_keypoints"] = num_keypoints

            if "segmentation" in annotation:
                coco_annotation["segmentation"] = annotation["segmentation"]

            coco_annotations.append(coco_annotation)

    logger.info(f"Conversion finished, num images: {len(coco_images)}, num annotations: {len(coco_annotations)}")

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file for efg.",
    }
    coco_dict = {
        "info": info,
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
        "licenses": None,
    }
    return coco_dict


def convert_to_coco_json(dataset_name, output_file, allow_cached=True):
    """
    Converts dataset into COCO format and saves it to a json file.
    dataset_name must be registered in DatasetCatalog and in efg's standard format.
    Args:
        dataset_name:
            reference from the config file to the catalogs
            must be registered in DatasetCatalog and in efg's standard format
        output_file: path of json file that will be saved to
        allow_cached: if json file is already present then skip conversion
    """

    # TODO: The dataset or the conversion script *may* change,
    # a checksum would be useful for validating the cached data

    PathManager.mkdirs(os.path.dirname(output_file))
    with file_lock(output_file):
        if PathManager.exists(output_file) and allow_cached:
            logger.info(f"Cached annotations in COCO format already exist: {output_file}")
        else:
            logger.info(f"Converting dataset annotations in '{dataset_name}' to COCO format ...)")
            coco_dict = convert_to_coco_dict(dataset_name)

            with PathManager.open(output_file, "w") as json_file:
                logger.info(f"Caching annotations in COCO format: {output_file}")
                json.dump(coco_dict, json_file)


def filter_images_with_only_crowd_annotations(dataset_dicts):
    """
    Filter out images with none annotations or only crowd annotations
    (i.e., images without non-crowd annotations).
    A common training-time preprocessing on COCO dataset.

    Args:
        dataset_dicts (list[dict]): annotations in cvpods Dataset format.

    Returns:
        list[dict]: the same format, but filtered.
    """
    num_before = len(dataset_dicts)

    def valid(anns):
        for ann in anns:
            if ann.get("iscrowd", 0) == 0:
                return True
        return False

    dataset_dicts = [x for x in dataset_dicts if valid(x["annotations"])]
    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info(f"Removed {num_before - num_after} images with no usable annotations. {num_after} images left.")
    return dataset_dicts


def print_instances_class_histogram(dataset_dicts, class_names):
    """
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,), dtype=np.int)
    for entry in dataset_dicts:
        annos = entry["annotations"]
        classes = [x["category_id"] for x in annos if not x.get("iscrowd", 0)]
        histogram += np.histogram(classes, bins=hist_bins)[0]

    N_COLS = min(6, len(class_names) * 2)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + ".."
        return x

    data = list(itertools.chain(*[[short_name(class_names[i]), int(v)] for i, v in enumerate(histogram)]))
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    if num_classes > 1:
        data.extend(["total", total_num_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instances"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    log_first_n(
        logging.INFO,
        "Distribution of instances among all {} categories:\n".format(num_classes) + colored(table, "cyan"),
        key="message",
    )
