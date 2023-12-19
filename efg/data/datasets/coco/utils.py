import logging

import numpy as np

import pycocotools.mask as mask_util

import torch

from efg.data.structures.boxes import Boxes, BoxMode
from efg.data.structures.instances import Instances
from efg.data.structures.keypoints import Keypoints
from efg.data.structures.masks import BitMasks, PolygonMasks, polygons_to_bitmask
from efg.data.structures.rotated_boxes import RotatedBoxes


def transform_instance_annotations(annotation, transforms, image_size):
    """
    Apply transforms to box and segmentation annotations of a single instance.
    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.
    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList or list[Transform]):
        image_size (tuple): the height, width of the transformed image
    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    # if isinstance(transforms, (tuple, list)):
    #     transforms = T.TransformList(transforms)

    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # clip transformed bbox to image size
    bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
    annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if "segmentation" in annotation:
        # each instance contains 1 or more polygons
        segm = annotation["segmentation"]
        if isinstance(segm, list):
            # polygons
            polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
            annotation["segmentation"] = [p.reshape(-1) for p in transforms.apply_polygons(polygons)]
        elif isinstance(segm, dict):
            # RLE
            mask = mask_util.decode(segm)
            mask = transforms.apply_segmentation(mask)
            assert tuple(mask.shape[:2]) == image_size
            annotation["segmentation"] = mask
        else:
            raise ValueError(
                "Cannot transform segmentation of type '{}'!"
                "Supported types are: polygons as list[list[float] or ndarray],"
                " COCO-style RLE as a dict.".format(type(segm))
            )

    return annotation


def transform_proposals(dataset_dict, image_shape, transforms, min_box_side_len, proposal_topk):
    """
    Apply transformations to the proposals in dataset_dict, if any.

    Args:
        dataset_dict (dict): a dict read from the dataset, possibly
            contains fields "proposal_boxes", "proposal_objectness_logits", "proposal_bbox_mode"
        image_shape (tuple): height, width
        transforms (TransformList):
        min_box_side_len (int): keep proposals with at least this size
        proposal_topk (int): only keep top-K scoring proposals

    The input dict is modified in-place, with abovementioned keys removed. A new
    key "proposals" will be added. Its value is an `Instances`
    object which contains the transformed proposals in its field
    "proposal_boxes" and "objectness_logits".
    """
    if "proposal_boxes" in dataset_dict:
        # Transform proposal boxes
        boxes = transforms.apply_box(
            BoxMode.convert(
                dataset_dict.pop("proposal_boxes"),
                dataset_dict.pop("proposal_bbox_mode"),
                BoxMode.XYXY_ABS,
            )
        )
        boxes = Boxes(boxes)
        objectness_logits = torch.as_tensor(dataset_dict.pop("proposal_objectness_logits").astype("float32"))

        boxes.clip(image_shape)
        keep = boxes.nonempty(threshold=min_box_side_len)
        boxes = boxes[keep]
        objectness_logits = objectness_logits[keep]

        proposals = Instances(image_shape)
        proposals.proposal_boxes = boxes[:proposal_topk]
        proposals.objectness_logits = objectness_logits[:proposal_topk]
        dataset_dict["proposals"] = proposals


def annotations_to_instances(annos, image_size, mask_format="polygon"):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    boxes = target.gt_boxes = Boxes(boxes)
    boxes.clip(image_size)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        if mask_format == "polygon":
            masks = PolygonMasks(segms)
        else:
            assert mask_format == "bitmask", mask_format
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    # polygon
                    masks.append(polygons_to_bitmask(segm, *image_size))
                elif isinstance(segm, dict):
                    # COCO RLE
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(segm.ndim)
                    # mask array
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict, or a full-image segmentation mask "
                        "as a 2D ndarray.".format(type(segm))
                    )
            # torch.from_numpy does not support array with negative stride.
            masks = BitMasks(torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks]))
        target.gt_masks = masks

    if len(annos) and "keypoints" in annos[0]:
        kpts = np.array([obj.get("keypoints", []) for obj in annos])  # (N, K, 3)
        # Set all out-of-boundary points to "unlabeled"
        kpts_xy = kpts[:, :, :2]
        inside = (kpts_xy >= np.array([0, 0])) & (kpts_xy <= np.array(image_size[::-1]))
        inside = inside.all(axis=2)
        kpts[:, :, :2] = kpts_xy
        kpts[:, :, 2][~inside] = 0
        target.gt_keypoints = Keypoints(kpts)

    return target


def annotations_to_instances_rotated(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.
    Compared to `annotations_to_instances`, this function is for rotated boxes only

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            Containing fields "gt_boxes", "gt_classes",
            if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [obj["bbox"] for obj in annos]
    target = Instances(image_size)
    boxes = target.gt_boxes = RotatedBoxes(boxes)
    boxes.clip(image_size)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    return target


def filter_empty_instances(instances, by_box=True, by_mask=True):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty())
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    # TODO: can also filter visible keypoints

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x
    return instances[m]


def create_keypoint_hflip_indices(dataset_names, meta):
    """
    Args:
        dataset_names (list[str]): list of dataset names
    Returns:
        ndarray[int]: a vector of size=#keypoints, storing the
        horizontally-flipped keypoint indices.
    """

    check_metadata_consistency("keypoint_names", dataset_names, meta)
    check_metadata_consistency("keypoint_flip_map", dataset_names, meta)

    names = meta["keypoint_names"]
    # TODO flip -> hflip
    flip_map = dict(meta["keypoint_flip_map"])
    flip_map.update({v: k for k, v in flip_map.items()})
    flipped_names = [i if i not in flip_map else flip_map[i] for i in names]
    flip_indices = [names.index(i) for i in flipped_names]
    return np.asarray(flip_indices)


# def gen_crop_transform_with_instance(crop_size, image_size, instance):
#     """
#     Generate a CropTransform so that the cropping region contains
#     the center of the given instance.
#
#     Args:
#         crop_size (tuple): h, w in pixels
#         image_size (tuple): h, w
#         instance (dict): an annotation dict of one instance, in cvpods's
#             dataset format.
#     """
#     crop_size = np.asarray(crop_size, dtype=np.int32)
#     bbox = BoxMode.convert(instance["bbox"], instance["bbox_mode"],
#                            BoxMode.XYXY_ABS)
#     center_yx = (bbox[1] + bbox[3]) * 0.5, (bbox[0] + bbox[2]) * 0.5
#
#     assert (image_size[0] >= center_yx[0] and image_size[1] >= center_yx[1]
#             ), "The annotation bounding box is outside of the image!"
#     assert (image_size[0] >= crop_size[0] and image_size[1] >= crop_size[1]
#             ), "Crop size is larger than image size!"
#
#     min_yx = np.maximum(np.ceil(center_yx).astype(np.int32) - crop_size, 0)
#     max_yx = np.maximum(np.asarray(image_size, dtype=np.int32) - crop_size, 0)
#     max_yx = np.minimum(max_yx, np.floor(center_yx).astype(np.int32))
#
#     y0 = np.random.randint(min_yx[0], max_yx[0] + 1)
#     x0 = np.random.randint(min_yx[1], max_yx[1] + 1)
#     return T.CropTransform(x0, y0, crop_size[1], crop_size[0])


def check_metadata_consistency(key, dataset_names, meta):
    """
    Check that the datasets have consistent metadata.

    Args:
        key (str): a metadata key
        dataset_names (list[str]): a list of dataset names

    Raises:
        AttributeError: if the key does not exist in the metadata
        ValueError: if the given datasets do not have the same metadata values defined by key
    """
    if len(dataset_names) == 0:
        return
    logger = logging.getLogger(__name__)
    entries_per_dataset = [meta.get(key) for d in dataset_names]
    for idx, entry in enumerate(entries_per_dataset):
        if entry != entries_per_dataset[0]:
            logger.error("Metadata '{}' for dataset '{}' is '{}'".format(key, dataset_names[idx], str(entry)))
            logger.error(
                "Metadata '{}' for dataset '{}' is '{}'".format(key, dataset_names[0], str(entries_per_dataset[0]))
            )
            raise ValueError("Datasets have different metadata '{}'!".format(key))


def check_sample_valid(args):
    if args["sample_style"] == "range":
        assert len(args["short_edge_length"]) == 2, (
            f"more than 2 ({len(args['short_edge_length'])}) " "short_edge_length(s) are provided for ranges"
        )
