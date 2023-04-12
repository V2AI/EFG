import numpy as np
from omegaconf import OmegaConf

from torch import nn

try:
    import spconv.pytorch as spconv
    from spconv.pytorch import SparseConv3d, SubMConv3d
except:
    import spconv
    from spconv import SparseConv3d, SubMConv3d

from efg.modeling.common.batch_norm import get_activation, get_norm

from ..registry import BACKBONES
from .backbone import Backbone, ShapeSpec


def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out


def build_norm_layer(cfg, num_features, postfix=""):
    """Build normalization layer
    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.
    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    norm_cfg = {
        # format: layer_type: (abbreviation, module)
        "BN": ("bn", nn.BatchNorm2d),
        "BN1d": ("bn1d", nn.BatchNorm1d),
        "GN": ("gn", nn.GroupNorm),
    }

    assert isinstance(cfg, dict) and "type" in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")
    if layer_type not in norm_cfg:
        raise KeyError("Unrecognized norm type {}".format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop("requires_grad", True)
    cfg_.setdefault("eps", 1e-5)
    if layer_type != "GN":
        layer = norm_layer(num_features, **cfg_)
        # if layer_type == 'SyncBN':
        #     layer._specify_ddp_gpu_num(1)
    else:
        assert "num_groups" in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer


class SparseBasicStem(spconv.SparseModule):
    def __init__(self, in_channels=16, out_channels=32, stem_width=32, norm="BN1d", activation=None, indice_key=None):
        super(SparseBasicStem, self).__init__()

        self.out_channels = out_channels

        self.conv1 = spconv.SparseSequential(
            SparseConv3d(in_channels, stem_width, 3, 2, padding=1, bias=False),
            get_norm(norm, stem_width),
            get_activation(activation),
            SubMConv3d(stem_width, stem_width, 3, padding=1, bias=False, indice_key=indice_key),
            get_norm(norm, stem_width),
            get_activation(activation),
            SubMConv3d(stem_width, out_channels, 3, padding=1, bias=False, indice_key=indice_key),
            get_norm(norm, out_channels),
            get_activation(activation),
        )

    def forward(self, x):
        return self.conv1(x)

    @property
    def stride(self):
        return 2  # = stride 2 conv


class SparseResNetBlockBase(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, stride):
        super(SparseResNetBlockBase, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        # freeze BN statistics here
        return self


class SparseBasicResBlock(SparseResNetBlockBase):
    def __init__(self, in_channels=32, out_channels=64, stride=1, norm="BN1d", activation=None, indice_key=None):
        super(SparseBasicResBlock, self).__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = spconv.SparseSequential(
                SparseConv3d(in_channels, out_channels, 3, padding=1, stride=stride, bias=False),
                get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        self.activation = get_activation(activation)

        is_subm = stride == 1
        self.conv = spconv.SparseSequential(
            SparseConv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            if not is_subm
            else SubMConv3d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key
            ),
            get_norm(norm, out_channels),
            get_activation(activation),
            SubMConv3d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key
            ),
            get_norm(norm, out_channels),
        )

        for layer in [self.conv, self.shortcut]:
            if layer is not None:
                # weight_init.c2_msra_fill(layer)
                pass

    def forward(self, x):
        out = self.conv(x)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out = replace_feature(out, out.features + shortcut.features)
        out = replace_feature(out, self.activation(out.features))

        return out


class SparseBottleneckBlock(SparseResNetBlockBase):
    def __init__(
        self, in_channels, out_channels, bottleneck_channels, stride=1, norm="BN1d", activation=None, indice_key=None
    ):
        super(SparseBottleneckBlock, self).__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = spconv.SparseSequential(
                SparseConv3d(in_channels, out_channels, 3, padding=1, stride=stride, bias=False),
                get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        self.activation = get_activation(activation)

        stride_1x1, stride_3x3 = (1, stride)

        is_subm = stride_3x3 == 1
        self.conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels, bottleneck_channels, kernel_size=1, stride=stride_1x1, bias=False, indice_key=indice_key
            ),
            get_norm(norm, bottleneck_channels),
            get_activation(activation),
            SparseConv3d(
                bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride_3x3, padding=1, bias=False
            )
            if not is_subm
            else SubMConv3d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=3,
                stride=stride_3x3,
                padding=1,
                bias=False,
                indice_key=indice_key,
            ),
            get_norm(norm, bottleneck_channels),
            get_activation(activation),
            spconv.SubMConv3d(bottleneck_channels, out_channels, kernel_size=1, bias=False, indice_key=indice_key),
            get_norm(norm, bottleneck_channels),
        )

        for layer in [self.conv, self.shortcut]:
            if layer is not None:  # shortcut can be None
                # weight_init.c2_msra_fill(layer)
                pass

    def forward(self, x):
        out = self.conv(x)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out = replace_feature(out, out.features + shortcut.features)
        out = replace_feature(out, self.activation(out.features))

        return out


def make_stage(block_class, num_blocks, first_stride, **kwargs):
    blocks = []
    for i in range(num_blocks):
        blocks.append(block_class(stride=first_stride if i == 0 else 1, **kwargs))
        kwargs["in_channels"] = kwargs["out_channels"]
    return blocks


class SparseResNet(Backbone):
    def __init__(self, stem, stages, out_channels=None, out_features=None, norm=None):
        super(SparseResNet, self).__init__()
        self.stem = stem
        self.out_channels = out_channels

        current_stride = self.stem.stride

        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stages_and_names = []
        for i, blocks in enumerate(stages):
            for block in blocks:
                assert isinstance(block, SparseResNetBlockBase), block
                # curr_channels = block.out_channels
            stage = spconv.SparseSequential(*blocks)
            name = "res" + str(i + 2)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            self._out_feature_strides[name] = current_stride = int(current_stride * np.prod([k.stride for k in blocks]))
            self._out_feature_channels[name] = blocks[-1].out_channels

        if out_features is None:
            out_features = [
                name,
            ]
        self._out_features = out_features
        assert len(self._out_features)

        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

        out_channels_multiplier = [6, 3, 2]
        for idx, out_feature in enumerate(self._out_features):
            channels = self._out_feature_channels[out_feature]
            out_layer = spconv.SparseSequential(
                SparseConv3d(channels, channels, (3, 1, 1), (2, 1, 1), padding=(1, 0, 0), bias=False),
                get_norm(norm, channels),
                nn.ReLU(),
            )
            self.add_module(out_feature + "_out", out_layer)
            self._out_feature_channels[out_feature] *= out_channels_multiplier[idx]

    def forward(self, voxel_features, coors, batch_size, input_shape):
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]
        coors = coors.int()

        # input: [41, 1280, 1280]
        x = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)

        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x

        # input: [21, 640, 640]
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x

        for out_feature in self._out_features:
            out = getattr(self, out_feature + "_out")(outputs[out_feature])
            out = out.dense()
            N, C, D, H, W = out.shape
            out = out.view(N, C * D, H, W)
            outputs[out_feature] = out

        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name])
            for name in self._out_features
        }


def build_sparse_resnet_backbone(config, in_channels):
    depth = config.depth
    stem_width = {
        18: 16,
        "18b": 24,
        "18c": 32,
        34: 16,
        "34b": 24,
        "34c": 32,
        50: 16,
    }[depth]

    norm = config.norm
    if not isinstance(norm, str):
        norm = OmegaConf.to_container(norm)

    activation = config.activation

    stem = SparseBasicStem(
        in_channels=in_channels,
        out_channels=config.stem_out_channels,
        norm=norm,
        activation=activation,
        stem_width=stem_width,
        indice_key="stem",
    )

    out_features = config.out_features
    num_groups = config.num_groups
    width_per_group = config.width_per_group
    bottleneck_channels = num_groups * width_per_group
    in_channels = config.stem_out_channels
    out_channels = config.res1_out_channels

    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        "18b": [2, 2, 2, 2],
        "18c": [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        "34b": [3, 4, 6, 3],
        "34c": [3, 4, 6, 3],
        50: [3, 4, 6, 3],
    }[depth]

    # Avoid creating variables without gradients
    # which consume extra memory and may cause allreduce to fail
    out_stage_idx = [{"res2": 2, "res3": 3, "res4": 4, "res5": 5, "linear": 5}[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)

    stages = []
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        # dilation = 1
        first_stride = 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "first_stride": first_stride,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
            "activation": activation,
            "indice_key": "res" + str(stage_idx),
        }
        # Use BasicBlock for R18 and R34.
        if depth in [18, "18b", "18c", 21, 34]:
            stage_kargs["block_class"] = SparseBasicResBlock
        else:
            stage_kargs["bottleneck_channels"] = bottleneck_channels
            # stage_kargs["stride_in_1x1"] = stride_in_1x1
            # stage_kargs["dilation"] = dilation
            # stage_kargs["num_groups"] = num_groups
            stage_kargs["block_class"] = SparseBottleneckBlock

        blocks = make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2

        stages.append(blocks)

    return SparseResNet(stem, stages, out_features=out_features, norm=norm)


"""Legacy code for Sparse ResNet"""


def conv3x3(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """3x3 convolution with padding"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )


def conv1x1(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """1x1 convolution"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm="BN1d",
        downsample=None,
        indice_key=None,
    ):
        super(SparseBasicBlock, self).__init__()

        bias = norm is not None

        self.conv1 = conv3x3(inplanes, planes, stride, indice_key=indice_key, bias=bias)
        self.bn1 = get_norm(norm, planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, indice_key=indice_key, bias=bias)
        self.bn2 = get_norm(norm, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


@BACKBONES.register()
class SpMiddleResNetFHD(Backbone):
    def __init__(
        self,
        num_input_features=128,
        out_features=[
            "res3",
        ],
        norm="BN1d",
    ):
        super(SpMiddleResNetFHD, self).__init__()

        # input: # [1600, 1200, 41]
        self.conv_input = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, bias=False, indice_key="res0"),
            get_norm(norm, 16),
            nn.ReLU(inplace=True),
        )

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm=norm, indice_key="res0"),
            SparseBasicBlock(16, 16, norm=norm, indice_key="res0"),
        )

        self.conv2 = spconv.SparseSequential(
            SparseConv3d(16, 32, 3, 2, padding=1, bias=False),  # [1600, 1200, 41] -> [800, 600, 21]
            get_norm(norm, 32),
            nn.ReLU(inplace=True),
            SparseBasicBlock(32, 32, norm=norm, indice_key="res1"),
            SparseBasicBlock(32, 32, norm=norm, indice_key="res1"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv3d(32, 64, 3, 2, padding=1, bias=False),  # [800, 600, 21] -> [400, 300, 11]
            get_norm(norm, 64),
            nn.ReLU(inplace=True),
            SparseBasicBlock(64, 64, norm=norm, indice_key="res2"),
            SparseBasicBlock(64, 64, norm=norm, indice_key="res2"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv3d(64, 128, 3, 2, padding=[0, 1, 1], bias=False),  # [400, 300, 11] -> [200, 150, 5]
            get_norm(norm, 128),
            nn.ReLU(inplace=True),
            SparseBasicBlock(128, 128, norm=norm, indice_key="res3"),
            SparseBasicBlock(128, 128, norm=norm, indice_key="res3"),
        )

        self.extra_conv = spconv.SparseSequential(
            SparseConv3d(128, 128, (3, 1, 1), (2, 1, 1), bias=False),  # [200, 150, 5] -> [200, 150, 2]
            get_norm(norm, 128),
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size, input_shape):
        # input: # [41, 1600, 1408]
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]

        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)

        x = self.conv_input(ret)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        ret = self.extra_conv(x_conv4)
        ret = ret.dense()
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        return ret
