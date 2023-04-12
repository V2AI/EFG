# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging

import numpy as np

import torch
from torch import nn

from efg.data.structures.shape_spec import ShapeSpec
from efg.modeling.common import weight_init
from efg.modeling.common.batch_norm import FrozenBatchNorm2d, get_activation, get_norm
from efg.modeling.common.blocks import Conv2d
from efg.modeling.operators.deform_conv import DeformConv, ModulatedDeformConv

from .backbone import Backbone

__all__ = [
    "ResNetBlockBase",
    "BottleneckBlock",
    "AVDBottleneckBlock",
    "DeformBottleneckBlock",
    "BasicStem",
    "ResNet",
    "make_stage",
    "build_resnet_backbone",
]


class ResNetBlockBase(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            in_channels (int):
            out_channels (int):
            stride (int):
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self


class BasicBlock(ResNetBlockBase):
    def __init__(self, in_channels, out_channels, *, stride=1, norm="BN", activation=None, **kwargs):
        """
        The standard block type for ResNet18 and ResNet34.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first conv.
            norm (str or callable): A callable that takes the number of
                channels and returns a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        self.activation = get_activation(activation)

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = self.activation(out)
        return out


class BottleneckBlock(ResNetBlockBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        activation=None,
        stride_in_1x1=False,
        dilation=1,
    ):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
            stride_in_1x1 (bool): when stride==2, whether to put stride in the
                first 1x1 convolution or the bottleneck 3x3 convolution.
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.activation = get_activation(activation)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        # Zero-initialize the last normalization in each residual branch,
        # so that at the beginning, the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # See Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
        # "For BN layers, the learnable scaling coefficient γ is initialized
        # to be 1, except for each residual block's last BN
        # where γ is initialized to be 0."

        # nn.init.constant_(self.conv3.norm.weight, 0)
        # TODO this somehow hurts performance when training GN models from scratch.
        # Add it as an option when we need to use this code to train a backbone.

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.activation(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = self.activation(out)

        return out


class AVDBottleneckBlock(BottleneckBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        activation=None,
        stride_in_1x1=False,
        dilation=1,
        avd=False,
        avg_down=False,
        radix=1,
        bottleneck_width=64,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            bottleneck_channels=bottleneck_channels,
            stride=stride,
            num_groups=num_groups,
            norm=norm,
            activation=activation,
            stride_in_1x1=stride_in_1x1,
            dilation=dilation,
        )

        self.avd = avd and (stride > 1)
        self.avg_down = avg_down
        self.radix = radix

        cardinality = num_groups
        group_width = int(bottleneck_channels * (bottleneck_width / 64.0)) * cardinality

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        if in_channels != out_channels and self.avg_down:
            assert self.shortcut is not None

            self.shortcut_avgpool = nn.AvgPool2d(
                kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False
            )
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
                norm=get_norm(norm, out_channels),
            )

        if self.radix > 1:
            from .splat import SplAtConv2d

            self.conv2 = SplAtConv2d(
                group_width,
                group_width,
                kernel_size=3,
                stride=1 if self.avd else stride_3x3,
                padding=dilation,
                dilation=dilation,
                groups=cardinality,
                bias=False,
                radix=self.radix,
                norm=norm,
            )
        else:
            assert hasattr(self, "conv2")

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)

        if self.radix > 1:
            for layer in [self.conv1, self.conv3, self.shortcut]:
                if layer is not None:  # shortcut can be None
                    weight_init.c2_msra_fill(layer)
        else:
            for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
                if layer is not None:  # shortcut can be None
                    weight_init.c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)

        if self.radix > 1:
            out = self.conv2(out)
        else:
            out = self.conv2(out)
            out = self.activation(out)

        if self.avd:
            out = self.avd_layer(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            if self.avg_down:
                x = self.shortcut_avgpool(x)
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = self.activation(out)

        return out


class DeformBottleneckBlock(ResNetBlockBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        activation=None,
        stride_in_1x1=False,
        dilation=1,
        deform_modulated=False,
        deform_num_groups=1,
    ):
        """
        Similar to :class:`BottleneckBlock`, but with deformable conv in the 3x3 convolution.
        """
        super().__init__(in_channels, out_channels, stride)
        self.deform_modulated = deform_modulated

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.activation = get_activation(activation)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        if deform_modulated:
            deform_conv_op = ModulatedDeformConv
            # offset channels are 2 or 3 (if with modulated) * kernel_size * kernel_size
            offset_channels = 27
        else:
            deform_conv_op = DeformConv
            offset_channels = 18

        self.conv2_offset = Conv2d(
            bottleneck_channels,
            offset_channels * deform_num_groups,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            dilation=dilation,
        )
        self.conv2 = deform_conv_op(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            deformable_groups=deform_num_groups,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        nn.init.constant_(self.conv2_offset.weight, 0)
        nn.init.constant_(self.conv2_offset.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)

        if self.deform_modulated:
            offset_mask = self.conv2_offset(out)
            offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((offset_x, offset_y), dim=1)
            mask = mask.sigmoid()
            out = self.conv2(out, offset, mask)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.activation(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = self.activation(out)
        return out


def make_stage(block_class, num_blocks, first_stride, **kwargs):
    """
    Create a resnet stage by creating many blocks.

    Args:
        block_class (class): a subclass of ResNetBlockBase
        num_blocks (int):
        first_stride (int): the stride of the first block. The other blocks will have stride=1.
            A `stride` argument will be passed to the block constructor.
        kwargs: other arguments passed to the block constructor.

    Returns:
        list[nn.Module]: a list of block module.
    """
    blocks = []
    for i in range(num_blocks):
        blocks.append(block_class(stride=first_stride if i == 0 else 1, **kwargs))
        kwargs["in_channels"] = kwargs["out_channels"]
    return blocks


class BasicStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, norm="BN", activation=None, deep_stem=False, stem_width=32):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
        """
        super().__init__()
        self.deep_stem = deep_stem

        if self.deep_stem:
            self.conv1_1 = Conv2d(
                3,
                stem_width,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                norm=get_norm(norm, stem_width),
            )
            self.conv1_2 = Conv2d(
                stem_width,
                stem_width,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm=get_norm(norm, stem_width),
            )
            self.conv1_3 = Conv2d(
                stem_width,
                stem_width * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm=get_norm(norm, stem_width * 2),
            )
            for layer in [self.conv1_1, self.conv1_2, self.conv1_3]:
                if layer is not None:
                    weight_init.c2_msra_fill(layer)
        else:
            self.conv1 = Conv2d(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
            weight_init.c2_msra_fill(self.conv1)

        self.activation = get_activation(activation)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        if self.deep_stem:
            x = self.conv1_1(x)
            x = self.activation(x)
            x = self.conv1_2(x)
            x = self.activation(x)
            x = self.conv1_3(x)
            x = self.activation(x)
        else:
            x = self.conv1(x)
            x = self.activation(x)
        x = self.max_pool(x)
        return x

    @property
    def out_channels(self):
        if self.deep_stem:
            return self.conv1_3.out_channels
        else:
            return self.conv1.out_channels

    @property
    def stride(self):
        return 4  # = stride 2 conv -> stride 2 max pool


class ResNet(Backbone):
    def __init__(self, stem, stages, num_classes=None, out_features=None, zero_init_residual=False):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[ResNetBlock]]): several (typically 4) stages,
                each contains multiple :class:`ResNetBlockBase`.
            num_classes (None or int): if None, will not perform classification.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
        """
        super(ResNet, self).__init__()
        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stages_and_names = []
        for i, blocks in enumerate(stages):
            for block in blocks:
                assert isinstance(block, ResNetBlockBase), block
                curr_channels = block.out_channels
            stage = nn.Sequential(*blocks)
            name = "res" + str(i + 2)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            self._out_feature_strides[name] = current_stride = int(current_stride * np.prod([k.stride for k in blocks]))
            self._out_feature_channels[name] = blocks[-1].out_channels

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            nn.init.normal_(self.linear.weight, std=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckBlock):
                    nn.init.constant_(m.conv3.norm.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.conv2.norm.weight, 0)

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name])
            for name in self._out_features
        }


def build_resnet_backbone(config, input_shape):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    depth = config.model.resnets.depth
    stem_width = {18: 32, 34: 32, 50: 32, 101: 64, 152: 64, 200: 64, 269: 64}[depth]
    deep_stem = config.model.resnets.deep_stem

    if not deep_stem:
        assert getattr(config.model.resnets, "radix", 1) <= 1, "config.model.resnets.radix: {} > 1".format(
            config.model.resnets.radix
        )

    # need registration of new blocks/stems?
    norm = config.model.resnets.norm
    activation = config.model.resnets.activation
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=config.model.resnets.stem_out_channels,
        norm=norm,
        activation=activation,
        deep_stem=deep_stem,
        stem_width=stem_width,
    )
    freeze_at = config.model.backbone.freeze_at

    if freeze_at >= 1:
        for p in stem.parameters():
            p.requires_grad = False
        stem = FrozenBatchNorm2d.convert_frozen_batchnorm(stem)

    out_features = config.model.resnets.out_features
    num_groups = config.model.resnets.num_groups
    width_per_group = config.model.resnets.width_per_group
    bottleneck_channels = num_groups * width_per_group
    in_channels = config.model.resnets.stem_out_channels
    out_channels = config.model.resnets.res2_out_channels
    stride_in_1x1 = config.model.resnets.stride_in_1x1
    res5_dilation = config.model.resnets.res5_dilation
    num_classes = config.model.resnets.num_classes
    zero_init_residual = config.model.resnets.zero_init_residual
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3],
        269: [3, 30, 48, 8],
    }[depth]

    # Avoid creating variables without gradients
    # which consume extra memory and may cause allreduce to fail
    out_stage_idx = [{"res2": 2, "res3": 3, "res4": 4, "res5": 5, "linear": 5}[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)
    # Apply Deformable Convolution in stages
    # Specify if apply deform_conv on Res2, Res3, Res4, Res5
    deform_on_per_stage = getattr(config.model.resnets, "deform_on_per_stage", [False] * (max_stage_idx - 1))

    if depth in [18, 34]:
        assert out_channels == 64, "Must set model.resnets.res2_out_channels = 64 for R18/R34"
        assert not any(deform_on_per_stage), "model.resnets.deform_on_per_stage unsupported for R18/R34"
        assert res5_dilation == 1, "Must set model.resnets.res5_dilation = 1 for R18/R34"
        assert num_groups == 1, "Must set model.resnets.num_groups = 1 for R18/R34"

    stages = []

    logger = logging.getLogger(__name__)
    # See efg/configs/base_detection_config.py for details
    if not stride_in_1x1 and "torchvision" not in config.model.weights:
        logger.warning("Using pretrain weight not from torchvision with model.resnets.stride_in_1x1 == False")
    elif stride_in_1x1 and "torchvision" in config.model.weights and config.model.weights:
        logger.warning("Using pretrain weight from torchvision with model.resnets.stride_in_1x1 == True")

    in_channels = 2 * stem_width if deep_stem else in_channels
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "first_stride": first_stride,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
            "activation": activation,
        }
        # Use BasicBlock for R18 and R34.
        if depth in [18, 34]:
            stage_kargs["block_class"] = BasicBlock
        else:
            stage_kargs["bottleneck_channels"] = bottleneck_channels
            stage_kargs["stride_in_1x1"] = stride_in_1x1
            stage_kargs["dilation"] = dilation
            stage_kargs["num_groups"] = num_groups
            if deform_on_per_stage[idx]:
                stage_kargs["block_class"] = DeformBottleneckBlock
                # Use True to use modulated deform_conv (DeformableV2);
                # Use False for DeformableV1.
                stage_kargs["deform_modulated"] = config.model.resnets.deform_modulated
                # Number of groups in deformable conv.
                stage_kargs["deform_num_groups"] = config.model.resnets.deform_num_groups
            elif hasattr(config.model.resnets, "RADIX"):
                # Radix in ResNeSt
                radix = config.model.resnets.radix
                # Apply avg after conv2 in the BottleBlock
                # When AVD=True, the STRIDE_IN_1X1 should be False
                avd = config.model.resnets.avd or (radix > 1)
                # Apply avg_down to the downsampling layer for residual path
                avg_down = config.model.resnets.avg_down or (radix > 1)
                # Bottleneck_width in ResNeSt
                bottleneck_width = config.model.resnets.bottleneck_width

                stage_kargs["block_class"] = AVDBottleneckBlock
                stage_kargs["avd"] = avd
                stage_kargs["avg_down"] = avg_down
                stage_kargs["radix"] = radix
                stage_kargs["bottleneck_width"] = bottleneck_width
            else:
                stage_kargs["block_class"] = BottleneckBlock
        blocks = make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2

        if freeze_at >= stage_idx:
            for block in blocks:
                block.freeze()
        stages.append(blocks)

    return ResNet(
        stem, stages, num_classes=num_classes, out_features=out_features, zero_init_residual=zero_init_residual
    )
