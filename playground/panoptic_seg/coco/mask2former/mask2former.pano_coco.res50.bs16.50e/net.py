from coco import COCODatasetV2

from backbone.swin import D2SwinTransformer
from mask2former import MaskFormer
from meta_arch.mask_former_head import MaskFormerHead
from meta_arch.per_pixel_baseline import PerPixelBaselineHead, PerPixelBaselinePlusHead
from optimizer import FullClipAdamW
from pixel_decoder.fpn import BasePixelDecoder
from pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder

__all__ = [
    "D2SwinTransformer", "COCODatasetV2", "MaskFormer", "MaskFormerHead", "PerPixelBaselineHead",
    "PerPixelBaselinePlusHead", "FullClipAdamW", "BasePixelDecoder", "MSDeformAttnPixelDecoder",
]


def build_model(self, config):
    model = MaskFormer(config)

    return model
