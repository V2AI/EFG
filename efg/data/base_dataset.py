from copy import deepcopy

from torch.utils.data import Dataset

from efg.utils.file_io import PathManager


class BaseDataset(Dataset):
    """Abstract class representing a pytorch-like Dataset.
    All subclasses should override:
        ``__len__`` that provides the size of the dataset,
        ``__getitem__`` that supports integer indexing in the range from 0 to length,
        ``_load_annotations`` that specfies how to access label files,
    """

    def __init__(self, config):
        """
        BaseDataset should have the following properties:
            * data_root (contains data and annotations)
            * transforms list
            * evaluators list

        Args:
            cfg (BaseConfig): config
            transforms (List[TransformGen]): list of transforms to get network input.
            is_train (bool): whether in training mode.
        """
        super(BaseDataset, self).__init__()
        self.config = config
        # all IO operation should be exected via PathManager
        self.path_manager = PathManager()

    def __getitem__(self, index):
        """Load data, apply transforms, converto to Instances."""
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def _load_annotations(self):
        raise NotImplementedError

    def _apply_transforms(self, image, annotations=None, **kwargs):
        """
        Apply a list of :class:`TransformGen` on the input image, and
        returns the transformed image and a list of transforms.

        We cannot simply create and return all transforms without
        applying it to the image, because a subsequent transform may
        need the output of the previous one.

        Args:
            transform_gens (list): list of :class:`TransformGen` instance to
                be applied.
            img (ndarray): uint8 or floating point images with 1 or 3 channels.
            annotations (list): annotations
        Returns:
            ndarray: the transformed image
            TransformList: contain the transforms that's used.
        """

        if isinstance(self.transforms, dict):
            dataset_dict = {}
            for key, tfms in self.transforms.items():
                img = deepcopy(image)
                annos = deepcopy(annotations)
                for tfm in tfms:
                    img, annos = tfm(img, annos, **kwargs)
                dataset_dict[key] = (img, annos)
            return dataset_dict, None
        else:
            for tfm in self.transforms:
                image, annotations = tfm(image, annotations, **kwargs)

            return image, annotations
