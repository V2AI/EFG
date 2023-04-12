# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Dict

from torch import nn

from efg.data.structures.shape_spec import ShapeSpec

from pixel_decoder.fpn import build_pixel_decoder
from transformer_decoder.maskformer_transformer_decoder import build_transformer_decoder


class MaskFormerHead(nn.Module):
    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "sem_seg_head" in k and not k.startswith(prefix + "predictor"):
                    newk = k.replace(prefix, prefix + "pixel_decoder.")
                    # logger.debug(f"{k} ==> {newk}")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    def __init__(self, config, input_shape: Dict[str, ShapeSpec]):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()

        # figure out in_channels to transformer predictor
        if config.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "transformer_encoder":
            transformer_predictor_in_channels = config.MODEL.SEM_SEG_HEAD.CONVS_DIM
        elif config.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "pixel_embedding":
            transformer_predictor_in_channels = config.MODEL.SEM_SEG_HEAD.MASK_DIM
        elif config.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "multi_scale_pixel_decoder":  # for maskformer2
            transformer_predictor_in_channels = config.MODEL.SEM_SEG_HEAD.CONVS_DIM
        else:
            transformer_predictor_in_channels = input_shape[config.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE].channels

        input_shape = {k: v for k, v in input_shape.items() if k in config.MODEL.SEM_SEG_HEAD.IN_FEATURES}
        sorted_input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in sorted_input_shape]
        # feature_strides = [v.stride for k, v in sorted_input_shape]
        # feature_channels = [v.channels for k, v in sorted_input_shape]

        self.ignore_value = config.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        self.common_stride = 4
        self.loss_weight = config.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT
        self.pixel_decoder = build_pixel_decoder(config, input_shape)
        self.predictor = build_transformer_decoder(config, transformer_predictor_in_channels, mask_classification=True)
        self.transformer_in_feature = config.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE

        self.num_classes = config.MODEL.SEM_SEG_HEAD.NUM_CLASSES

    def forward(self, features, mask=None):
        return self.layers(features, mask)

    def layers(self, features, mask=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
            features
        )
        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            predictions = self.predictor(multi_scale_features, mask_features, mask)
        else:
            if self.transformer_in_feature == "transformer_encoder":
                assert transformer_encoder_features is not None, "Please use the TransformerEncoderPixelDecoder."
                predictions = self.predictor(transformer_encoder_features, mask_features, mask)
            elif self.transformer_in_feature == "pixel_embedding":
                predictions = self.predictor(mask_features, mask_features, mask)
            else:
                predictions = self.predictor(features[self.transformer_in_feature], mask_features, mask)
        return predictions
