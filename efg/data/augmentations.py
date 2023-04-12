import inspect
import pprint
import sys
from abc import abstractmethod
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import pycocotools.mask as mask_util

import torch

from efg.data.registry import PROCESSORS
from efg.data.structures.boxes import BoxMode


def build_processors(pipelines):
    transforms = []
    for pipeline in pipelines:
        if isinstance(pipeline, dict):
            name, args = pipeline.copy().popitem()
            transform = PROCESSORS.get(name)(**args)
            transforms.append(transform)
        else:
            transform = PROCESSORS.get(pipeline)()
            transforms.append(transform)

    return transforms


class AugmentationBase:
    def _init(self, params=None):
        if params and isinstance(params, dict):
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    def _rand_range(self, low=1.0, high=None, size=None):
        """
        Uniform float random number between low and high.
        """
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return np.random.uniform(low, high, size)

    def __call__(self, data: np.ndarray, annotations: list = None, **kwargs):
        """
        Apply transform to the data and corresponding annotations (if exist).
        """
        raise NotImplementedError

    def __repr__(self):
        """
        Print Augmentation with hyper-params: "Augmentation(field1={self.field1}, field2={self.field2})"
        """
        try:
            sig = inspect.signature(self.__init__)
            classname = type(self).__name__
            argstr = []
            for name, param in sig.parameters.items():
                assert (
                    param.kind != param.VAR_POSITIONAL and param.kind != param.VAR_KEYWORD
                ), "The default __repr__ doesn't support *args or **kwargs"
                assert hasattr(
                    self, name
                ), f"Attribute {name} not found! Default __repr__ only works if attributes match the constructor."
                attr = getattr(self, name)
                default = param.default
                if default is attr:
                    continue
                attr_str = pprint.pformat(attr)
                if "\n" in attr_str:
                    # don't show it if pformat decides to use >1 lines
                    attr_str = "..."
                argstr.append("{}={}".format(name, attr_str))
            return "{}({})".format(classname, ", ".join(argstr))
        except AssertionError:
            return super().__repr__()


@PROCESSORS.register()
class NoOpAugmentation(AugmentationBase):
    def __call__(self, data, annotaions=None, **kwargs):
        return data, annotaions


class Augmentation2DBase(AugmentationBase):
    @abstractmethod
    def apply_image(self, img: np.ndarray):
        """
        Apply the transform on an image.
        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: image after apply the transformation.
        """

    @abstractmethod
    def apply_coords(self, coords: np.ndarray):
        """
        Apply the transform on coordinates.
        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is (x, y).
        Returns:
            ndarray: coordinates after apply the transformation.
        Note:
            The coordinates are not pixel indices. Coordinates inside an image of
            shape (H, W) are in range [0, W] or [0, H].
            This function should correctly transform coordinates outside the image as well.
        """

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply the transform on a full-image segmentation.
        By default will just perform "apply_image".
        Args:
            segmentation (ndarray): of shape HxW. The array should have integer
            or bool dtype.
        Returns:
            ndarray: segmentation after apply the transformation.
        """
        return self.apply_image(segmentation)

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        """
        Apply the transform on an axis-aligned box. By default will transform
        the corner points and use their minimum/maximum to create a new
        axis-aligned box. Note that this default may change the size of your
        box, e.g. after rotations.
        Args:
            box (ndarray): Nx4 floating point array of XYXY format in absolute
                coordinates.
        Returns:
            ndarray: box after apply the transformation.
        Note:
            The coordinates are not pixel indices. Coordinates inside an image of
            shape (H, W) are in range [0, W] or [0, H].
            This function does not clip boxes to force them inside the image.
            It is up to the application that uses the boxes to decide.
        """
        # Indexes of converting (x0, y0, x1, y1) box into 4 coordinates of
        # ([x0, y0], [x1, y0], [x0, y1], [x1, y1]).
        idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
        coords = np.asarray(box).reshape(-1, 4)[:, idxs].reshape(-1, 2)
        coords = self.apply_coords(coords).reshape((-1, 4, 2))
        minxy = coords.min(axis=1)
        maxxy = coords.max(axis=1)
        trans_boxes = np.concatenate((minxy, maxxy), axis=1)
        return trans_boxes

    def apply_polygons(self, polygons: list) -> list:
        """
        Apply the transform on a list of polygons, each represented by a Nx2
        array. By default will just transform all the points.
        Args:
            polygon (list[ndarray]): each is a Nx2 floating point array of
                (x, y) format in absolute coordinates.
        Returns:
            list[ndarray]: polygon after apply the transformation.
        Note:
            The coordinates are not pixel indices. Coordinates on an image of
            shape (H, W) are in range [0, W] or [0, H].
        """
        return [self.apply_coords(p) for p in polygons]

    @classmethod
    def register_type(cls, data_type: str, func: Optional[Callable] = None):
        """
        Register the given function as a handler that this transform will use
        for a specific data type.
        Args:
            data_type (str): the name of the data type (e.g., box)
            func (callable): takes a transform and a data, returns the
                transformed data.
        Examples:
        .. code-block:: python
            # call it directly
            def func(flip_transform, voxel_data):
                return transformed_voxel_data
            HFlipTransform.register_type("voxel", func)
            # or, use it as a decorator
            @HFlipTransform.register_type("voxel")
            def func(flip_transform, voxel_data):
                return transformed_voxel_data
            # ...
            transform = HFlipTransform(...)
            transform.apply_voxel(voxel_data)  # func will be called
        """
        if func is None:  # the decorator style

            def wrapper(decorated_func):
                assert decorated_func is not None
                cls.register_type(data_type, decorated_func)
                return decorated_func

            return wrapper

        assert callable(func), f"You can only register a callable to a Transform. Got {func} instead."
        argspec = inspect.getfullargspec(func)
        assert len(argspec.args) == 2, (
            "You can only register a function that takes two positional "
            "arguments to a Transform! Got a function with spec {}".format(str(argspec))
        )
        setattr(cls, "apply_" + data_type, func)

    def __call__(self, image, annotations=None, **kwargs):
        """
        Apply transfrom to images and annotations (if exist)
        """
        image_size = image.shape[:2]  # h, w
        image = self.apply_image(image)

        if annotations is not None:
            for annotation in annotations:
                if "bbox" in annotation:
                    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
                    # Note that bbox is 1d (per-instance bounding box)
                    annotation["bbox"] = self.apply_box([bbox])[0]
                    annotation["bbox_mode"] = BoxMode.XYXY_ABS

                if "segmentation" in annotation:
                    # each instance contains 1 or more polygons
                    segm = annotation["segmentation"]
                    if isinstance(segm, list):
                        # polygons
                        polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
                        annotation["segmentation"] = [p.reshape(-1) for p in self.apply_polygons(polygons)]
                    elif isinstance(segm, dict):
                        # RLE
                        mask = mask_util.decode(segm)
                        mask = self.apply_segmentation(mask)
                        assert tuple(mask.shape[:2]) == image_size
                        annotation["segmentation"] = mask
                    else:
                        raise ValueError(
                            "Cannot transform segmentation of type '{}'!"
                            "Supported types are: polygons as list[list[float] or ndarray],"
                            " COCO-style RLE as a dict.".format(type(segm))
                        )

                # For sem seg task
                if "sem_seg" in annotation:
                    sem_seg = annotation["sem_seg"]
                    if isinstance(sem_seg, np.ndarray):
                        sem_seg = self.apply_segmentation(sem_seg)
                        assert tuple(sem_seg.shape[:2]) == tuple(
                            image.shape[:2]
                        ), f"Image shape is {image.shape[:2]}, but sem_seg shape is {sem_seg.shape[:2]}."
                        annotation["sem_seg"] = sem_seg
                    else:
                        raise ValueError(
                            f"Cannot transform segmentation of type '{sem_seg}'! Supported type is ndarray."
                        )

        return image, annotations


class PadTransform(Augmentation2DBase):
    def __init__(self, x0: int, y0: int, x1: int, y1: int, pad_value: float = 0, seg_pad_value: int = 0):
        """
        Args:
            x0, y0: number of padded pixels on the left and top
            x1, y1: number of padded pixels on the right and bottom
            orig_w, orig_h: optional, original width and height.
                Needed to make this transform invertible.
            pad_value: the padding value to the image
            seg_pad_value: the padding value to the segmentation mask
        """
        super().__init__()
        self._init(locals())

    def apply_image(self, img):
        if img.ndim == 3:
            padding = ((self.y0, self.y1), (self.x0, self.x1), (0, 0))
        else:
            padding = ((self.y0, self.y1), (self.x0, self.x1))
        return np.pad(
            img,
            padding,
            mode="constant",
            constant_values=self.pad_value,
        )

    def apply_segmentation(self, img):
        if img.ndim == 3:
            padding = ((self.y0, self.y1), (self.x0, self.x1), (0, 0))
        else:
            padding = ((self.y0, self.y1), (self.x0, self.x1))
        return np.pad(
            img,
            padding,
            mode="constant",
            constant_values=self.seg_pad_value,
        )

    def apply_coords(self, coords):
        coords[:, 0] += self.x0
        coords[:, 1] += self.y0
        return coords


class CroppTransform(Augmentation2DBase):
    """
    Perform crop operations on images.
    """

    def __init__(self, x0: int, y0: int, w: int, h: int):
        """
        Args:
            x0, y0, w, h (int): crop the image(s) by img[y0:y0+h, x0:x0+w].
        """
        super().__init__()
        self._init(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Crop the image(s).
        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: cropped image(s).
        """
        if len(img.shape) <= 3:
            return img[self.y0 : self.y0 + self.h, self.x0 : self.x0 + self.w]
        else:
            return img[..., self.y0 : self.y0 + self.h, self.x0 : self.x0 + self.w, :]

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply crop transform on coordinates.
        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is (x, y).
        Returns:
            ndarray: cropped coordinates.
        """
        coords[:, 0] -= self.x0
        coords[:, 1] -= self.y0
        return coords

    def apply_polygons(self, polygons: list) -> list:
        """
        Apply crop transform on a list of polygons, each represented by a Nx2 array.
        It will crop the polygon with the box, therefore the number of points in the
        polygon might change.
        Args:
            polygon (list[ndarray]): each is a Nx2 floating point array of
                (x, y) format in absolute coordinates.
        Returns:
            ndarray: cropped polygons.
        """
        import shapely.geometry as geometry

        # Create a window that will be used to crop
        crop_box = geometry.box(self.x0, self.y0, self.x0 + self.w, self.y0 + self.h).buffer(0.0)

        cropped_polygons = []

        for polygon in polygons:
            polygon = geometry.Polygon(polygon).buffer(0.0)
            # polygon must be valid to perform intersection.
            assert polygon.is_valid, polygon
            cropped = polygon.intersection(crop_box)
            if cropped.is_empty:
                continue
            if not isinstance(cropped, geometry.collection.BaseMultipartGeometry):
                cropped = [cropped]
            # one polygon may be cropped to multiple ones
            for poly in cropped:
                # It could produce lower dimensional objects like lines or
                # points, which we want to ignore
                if not isinstance(poly, geometry.Polygon) or not poly.is_valid:
                    continue
                coords = np.asarray(poly.exterior.coords)
                # NOTE This process will produce an extra identical vertex at
                # the end. So we remove it. This is tested by
                # `tests/test_data_transform.py`
                cropped_polygons.append(coords[:-1])
        return [self.apply_coords(p) for p in cropped_polygons]


class ResizeTransform(Augmentation2DBase):
    """
    Resize the image to a target size.
    """

    def __init__(self, h, w, new_h, new_w, interp):
        """
        Args:
            h, w (int): original image size
            new_h, new_w (int): new image size
            interp: PIL interpolation methods
        """
        # TODO decide on PIL vs opencv
        super().__init__()
        self._init(locals())

    def apply_image(self, img, interp=None):
        assert img.shape[:2] == (self.h, self.w)
        pil_image = Image.fromarray(img)
        interp_method = interp if interp is not None else self.interp
        pil_image = pil_image.resize((self.new_w, self.new_h), interp_method)
        ret = np.asarray(pil_image)
        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation


@PROCESSORS.register()
class RandomFlip(Augmentation2DBase):
    """
    Perform horizontal flip.
    """

    def __init__(self, prob=0.5, horizontal=True, vertical=False):
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        super().__init__()

        if horizontal and vertical:
            raise ValueError("Cannot do both horiz and vert. Please use two Flip instead.")
        if not horizontal and not vertical:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._init(locals())

    def __call__(self, image, annotations, **kwargs):
        h, w = image.shape[:2]
        do = self._rand_range() < self.prob

        if self.horizontal:
            self.width = w
        else:
            self.height = h

        if do:
            return super().__call__(image, annotations, **kwargs)
        else:
            return image, annotations

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Flip the image(s).
        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: the flipped image(s).
        """
        if self.horizontal:
            tensor = torch.from_numpy(np.ascontiguousarray(img).copy())
            if len(tensor.shape) == 2:
                # For dimension of HxW.
                tensor = tensor.flip((-1))
            elif len(tensor.shape) > 2:
                # For dimension of HxWxC, NxHxWxC.
                tensor = tensor.flip((-2))
        else:
            tensor = torch.from_numpy(np.ascontiguousarray(img).copy())
            if len(tensor.shape) == 2:
                # For dimension of HxW.
                tensor = tensor.flip((-2))
            elif len(tensor.shape) > 2:
                # For dimension of HxWxC, NxHxWxC.
                tensor = tensor.flip((-3))

        return tensor.numpy()

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Flip the coordinates.
        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is (x, y).
        Returns:
            ndarray: the flipped coordinates.
        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, H - y)`, not
            `(W - 1 - x, H 1 - y)`.
        """
        if self.horizontal:
            coords[:, 0] = self.width - coords[:, 0]
        else:
            coords[:, 1] = self.height - coords[:, 1]

        return coords


@PROCESSORS.register()
class ResizeShortestEdge(ResizeTransform):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(self, short_edge_length, max_size=sys.maxsize, sample_style="range", interp=Image.BILINEAR):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
            interp: PIL interpolation method.
        """
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        self._init(locals())

    def __call__(self, img, annotations=None, **kwargs):
        h, w = img.shape[:2]

        if self.is_range:
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.short_edge_length)

        if size == 0:
            return img, annotations

        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)

        self.h, self.w, self.new_h, self.new_w = h, w, newh, neww
        return super().__call__(img, annotations)


@PROCESSORS.register()
class FixedSizeCrop(CroppTransform):
    """
    If `crop_size` is smaller than the input image size, then it uses a random crop of
    the crop size. If `crop_size` is larger than the input image size, then it pads
    the right and the bottom of the image to the crop size if `pad` is True, otherwise
    it returns the smaller image.
    """

    def __init__(self, crop_size: Tuple[int], pad: bool = True, pad_value: float = 128.0, seg_pad_value: int = 255):
        """
        Args:
            crop_size: target image (height, width).
            pad: if True, will pad images smaller than `crop_size` up to `crop_size`
            pad_value: the padding value to the image.
            seg_pad_value: the padding value to the segmentation mask.
        """
        self._init(locals())

    def __call__(self, image, annotations=None, **kwargs):
        input_size = image.shape[:2]
        output_size = self.crop_size

        # Add random crop if the image is scaled up.
        max_offset = np.subtract(input_size, output_size)
        max_offset = np.maximum(max_offset, 0)
        offset = np.multiply(max_offset, np.random.uniform(0.0, 1.0))
        offset = np.round(offset).astype(int)

        self.x0, self.y0, self.w, self.h = (
            offset[1],
            offset[0],
            output_size[1],
            output_size[0],
        )
        image, annotations = super().__call__(image, annotations, **kwargs)

        # prepare for padding if necessary
        input_size = image.shape[:2]
        # Add padding if the image is scaled down.
        pad_size = np.subtract(output_size, input_size)
        pad_size = np.maximum(pad_size, 0)
        return PadTransform(
            0,
            0,
            pad_size[1],
            pad_size[0],
            self.pad_value,
            self.seg_pad_value,
        )(
            image,
            annotations,
            **kwargs,
        )


@PROCESSORS.register()
class ResizeScale(ResizeTransform):
    """
    Takes target size as input and randomly scales the given target size between `min_scale`
    and `max_scale`. It then scales the input image such that it fits inside the scaled target
    box, keeping the aspect ratio constant.
    This implements the resize part of the Google's 'resize_and_crop' data augmentation:
    https://github.com/tensorflow/tpu/blob/master/models/official/detection/utils/input_utils.py#L127
    """

    def __init__(
        self, min_scale: float, max_scale: float, target_height: int, target_width: int, interp: int = Image.BILINEAR
    ):
        """
        Args:
            min_scale: minimum image scale range.
            max_scale: maximum image scale range.
            target_height: target image height.
            target_width: target image width.
            interp: image interpolation method.
        """
        self._init(locals())

    def __call__(self, image, annotaions=None, **kwargs):
        input_size = image.shape[:2]

        # Compute new target size given a scale.
        target_size = (self.target_height, self.target_width)
        scale = np.random.uniform(self.min_scale, self.max_scale)
        target_scale_size = np.multiply(target_size, scale)

        # Compute actual rescaling applied to input image and output size.
        output_scale = np.minimum(target_scale_size[0] / input_size[0], target_scale_size[1] / input_size[1])
        output_size = np.round(np.multiply(input_size, output_scale)).astype(int)

        self.h, self.w, self.new_h, self.new_w = input_size[0], input_size[1], output_size[0], output_size[1]
        return super().__call__(image, annotaions, **kwargs)


@PROCESSORS.register()
class Distortion(Augmentation2DBase):
    """
    Distort image w.r.t hue, saturation and exposure.
    """

    def __init__(self, hue, saturation, exposure, image_format):
        super().__init__()
        self.cvt_code = {
            "RGB": (cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2RGB),
            "BGR": (cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR),
        }[image_format]
        if saturation > 1.0:
            saturation /= 255.0  # in range [0, 1]
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: the distorted image(s).
        """
        dhue = np.random.uniform(low=-self.hue, high=self.hue)
        dsat = self._rand_scale(self.saturation)
        dexp = self._rand_scale(self.exposure)

        dtype = img.dtype
        img = cv2.cvtColor(img, self.cvt_code[0])
        img = np.asarray(img, dtype=np.float32) / 255.0

        img[:, :, 1] *= dsat
        img[:, :, 2] *= dexp
        H = img[:, :, 0] + dhue
        if dhue > 0:
            H[H > 1.0] -= 1.0
        else:
            H[H < 0.0] += 1.0
        img[:, :, 0] = H

        img = (img * 255).clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, self.cvt_code[1])
        img = np.asarray(img, dtype=dtype)

        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def _rand_scale(self, upper_bound):
        """
        Calculate random scaling factor.
        Args:
            upper_bound (float): range of the random scale.
        Returns:
            random scaling factor (float) whose range is
            from 1 / s to s .
        """
        scale = np.random.uniform(low=1, high=upper_bound)
        if np.random.rand() > 0.5:
            return scale
        return 1 / scale

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation


class BlendTransform(Augmentation2DBase):
    """
    Transforms pixel colors with PIL enhance functions.
    """

    def __init__(self, src_image: np.ndarray, src_weight: float, dst_weight: float):
        """
        Blends the input image (dst_image) with the src_image using formula:
        ``src_weight * src_image + dst_weight * dst_image``
        Args:
            src_image (ndarray): Input image is blended with this image
            src_weight (float): Blend weighting of src_image
            dst_weight (float): Blend weighting of dst_image
        """
        super().__init__()
        self._init(locals())

    def __call__(self, img, annotations=None, **kwargs):
        """
        Apply blend transform on the image(s).
        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
            interp (str): keep this option for consistency, perform blend would not
                require interpolation.
        Returns:
            ndarray: blended image(s).
        """
        if img.dtype == np.uint8:
            img = img.astype(np.float32)
            img = self.src_weight * self.src_image + self.dst_weight * img
            return np.clip(img, 0, 255).astype(np.uint8), annotations
        else:
            return self.src_weight * self.src_image + self.dst_weight * img, annotations


@PROCESSORS.register()
class RandomBrightness(BlendTransform):
    """
    Randomly transforms image brightness.
    Brightness intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce brightness
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase brightness
    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min, intensity_max, prob=1.0):
        """
        Args:
            intensity_min (float): Minimum augmentation.
            intensity_max (float): Maximum augmentation.
            prob (float): probability of transforms image brightness.
        """
        super().__init__()
        self._init(locals())

    def __call__(self, img, annotations=None, **kwargs):
        if self._rand_range() < self.prob:
            w = np.random.uniform(self.intensity_min, self.intensity_max)
            self.src_image, self.src_weight, self.dst_weight = 0, 1 - w, w
            return super().__call__(img, annotations)
        else:
            return img, annotations


@PROCESSORS.register()
class RandomSaturation(BlendTransform):
    """
    Randomly transforms image saturation.
    Saturation intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce saturation (make the image more grayscale)
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase saturation
    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min, intensity_max, prob=1.0):
        """
        Args:
            intensity_min (float): Minimum augmentation (1 preserves input).
            intensity_max (float): Maximum augmentation (1 preserves input).
            prob (float): probability of transforms image saturation.
        """
        super().__init__()
        self._set_attributes(locals())

    def __call__(self, img, annotations=None, **kwargs):
        if self._rand_range() < self.prob:
            assert img.shape[-1] == 3, "Saturation only works on RGB images"
            w = np.random.uniform(self.intensity_min, self.intensity_max)
            grayscale = img.dot([0.299, 0.587, 0.114])[:, :, np.newaxis]
            self.src_image, self.src_weight, self.dst_weight = grayscale, 1 - w, w
            return super().__call__(img, annotations)
        else:
            return img, annotations


@PROCESSORS.register()
class RandomLightning(BlendTransform):
    """
    Randomly transforms image color using fixed PCA over ImageNet.
    The degree of color jittering is randomly sampled via a normal distribution,
    with standard deviation given by the scale parameter.
    """

    def __init__(self, scale, prob=0.5):
        """
        Args:
            scale (float): Standard deviation of principal component weighting.
        """
        super().__init__()
        self._set_attributes(locals())
        self.eigen_vecs = np.array(
            [
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203],
            ]
        )
        self.eigen_vals = np.array([0.2175, 0.0188, 0.0045])

    def __call__(self, img, annotations=None, **kwargs):
        assert img.shape[-1] == 3, "Saturation only works on RGB/BGR images"
        if self._rand_range() < self.prob:
            weights = np.random.normal(scale=self.scale, size=3)
            self.src_image, self.src_weight, self.dst_weight = self.eigen_vecs.dot(weights * self.eigen_vals), 1, 1
            return super().__call__(img, annotations)
        else:
            return img, annotations


@PROCESSORS.register()
class RandomSwapChannels(Augmentation2DBase):
    """
    Randomly swap image channels.
    """

    def __init__(self, prob=0.5):
        super().__init__()
        self._set_attributes()

    def __call__(self, img, annotations=None, **kwargs):
        assert len(img.shape) > 2
        if self._rand_range() < self.prob:
            return img[..., np.random.permutation(3)], annotations
        else:
            return img, annotations
