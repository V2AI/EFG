# usage example:
# efg/experimental/voxel_detr/boxattn.waymo.sparse_resnet.2enc.2dec.3cat.grid01.bs48.epoch6.down8x.regionvit.refine

import copy

import torch
from torch import nn
from torch.nn import functional as F

from attention.attention2d import LayerNorm2d
from attention.attention_variants import AttentionWithRelPos
from timm.models.vision_transformer import Mlp


class R2LAttentionPlusFFN(nn.Module):
    def __init__(
        self,
        input_channels,
        dim_hidden,
        kernel_size,
        num_heads,
        mlp_ratio=1.0,
        qkv_bias=False,
        qk_scale=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_drop=0.0,
        drop=0.0,
        cls_attn=True,
    ):
        super().__init__()

        if not isinstance(kernel_size, (tuple, list)):
            kernel_size = [(kernel_size, kernel_size), (kernel_size, kernel_size), 0]
        self.kernel_size = kernel_size

        if cls_attn:
            self.norm0 = norm_layer(input_channels)
        else:
            self.norm0 = None

        self.norm1 = norm_layer(input_channels)
        self.attn = AttentionWithRelPos(
            input_channels,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_map_dim=(kernel_size[0][0], kernel_size[0][1]),
            num_cls_tokens=1,
        )
        self.norm2 = norm_layer(input_channels)
        self.mlp = Mlp(
            in_features=input_channels,
            hidden_features=int(dim_hidden * mlp_ratio),
            out_features=dim_hidden,
            act_layer=act_layer,
            drop=drop,
        )

        self.expand = (
            nn.Sequential(norm_layer(input_channels), act_layer(), nn.Linear(input_channels, dim_hidden))
            if input_channels != dim_hidden
            else None
        )

        self.linear = nn.Linear(dim_hidden, input_channels)

    def forward(self, xs):
        out, B, H, W, mask = xs
        cls_tokens = out[:, 0:1, ...]

        C = cls_tokens.shape[-1]
        cls_tokens = cls_tokens.reshape(B, -1, C)  # (N)x(H/sxW/s)xC

        if self.norm0 is not None:
            cls_tokens = cls_tokens + self.attn(self.norm0(cls_tokens))  # (N)x(H/sxK/s)xC

        # ks, stride, padding = self.kernel_size
        cls_tokens = cls_tokens.reshape(-1, 1, C)  # (NxH/sxK/s)x1xC

        out = torch.cat((cls_tokens, out[:, 1:, ...]), dim=1)
        tmp = out

        tmp = tmp + self.attn(self.norm1(tmp), patch_attn=True, mask=mask)
        identity = self.expand(tmp) if self.expand is not None else tmp
        tmp = identity + self.mlp(self.norm2(tmp))

        return self.linear(tmp)


class Projection(nn.Module):
    def __init__(self, input_channels, output_channels, act_layer, mode="sc"):
        super().__init__()
        tmp = []
        if "c" in mode:
            ks = 2 if "s" in mode else 1
            if ks == 2:
                stride = ks
                ks = ks + 1
                padding = ks // 2
            else:
                stride = ks
                padding = 0

            if input_channels == output_channels and ks == 1:
                tmp.append(nn.Identity())
            else:
                tmp.extend(
                    [
                        LayerNorm2d(input_channels),
                        act_layer(),
                    ]
                )
                tmp.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=output_channels,
                        kernel_size=ks,
                        stride=stride,
                        padding=padding,
                        groups=input_channels,
                    )
                )

        self.proj = nn.Sequential(*tmp)
        self.proj_cls = self.proj

    def forward(self, xs):
        cls_tokens, patch_tokens = xs
        # x: BxCxHxW
        cls_tokens = self.proj_cls(cls_tokens)
        patch_tokens = self.proj(patch_tokens)
        return cls_tokens, patch_tokens


def convert_to_flatten_layout(cls_tokens, patch_tokens, ws):
    """
    Convert the token layer in a flatten form, it will speed up the model.
    Furthermore, it also handle the case that if the size between regional tokens and local tokens are not consistent.
    """
    # padding if needed, and all paddings are happened at bottom and right.
    B, C, H, W = patch_tokens.shape
    _, _, H_ks, W_ks = cls_tokens.shape
    need_mask = False
    p_l, p_r, p_t, p_b = 0, 0, 0, 0
    if H % (H_ks * ws) != 0 or W % (W_ks * ws) != 0:
        p_l, p_r = 0, W_ks * ws - W
        p_t, p_b = 0, H_ks * ws - H
        patch_tokens = F.pad(patch_tokens, (p_l, p_r, p_t, p_b))
        need_mask = True

    B, C, H, W = patch_tokens.shape
    kernel_size = (H // H_ks, W // W_ks)
    tmp = F.unfold(patch_tokens, kernel_size=kernel_size, stride=kernel_size, padding=(0, 0))  # Nx(Cxksxks)x(H/sxK/s)
    patch_tokens = (
        tmp.transpose(1, 2).reshape(-1, C, kernel_size[0] * kernel_size[1]).transpose(-2, -1)
    )  # (NxH/sxK/s)x(ksxks)xC

    if need_mask:
        BH_sK_s, ksks, C = patch_tokens.shape
        H_s, W_s = H // ws, W // ws
        mask = torch.ones(BH_sK_s // B, 1 + ksks, 1 + ksks, device=patch_tokens.device, dtype=torch.float)
        right = torch.zeros(1 + ksks, 1 + ksks, device=patch_tokens.device, dtype=torch.float)
        tmp = torch.zeros(ws, ws, device=patch_tokens.device, dtype=torch.float)
        tmp[0 : (ws - p_r), 0 : (ws - p_r)] = 1.0
        tmp = tmp.repeat(ws, ws)
        right[1:, 1:] = tmp
        right[0, 0] = 1
        right[0, 1:] = torch.tensor([1.0] * (ws - p_r) + [0.0] * p_r).repeat(ws).to(right.device)
        right[1:, 0] = torch.tensor([1.0] * (ws - p_r) + [0.0] * p_r).repeat(ws).to(right.device)
        bottom = torch.zeros_like(right)
        bottom[0 : ws * (ws - p_b) + 1, 0 : ws * (ws - p_b) + 1] = 1.0
        bottom_right = copy.deepcopy(right)
        bottom_right[0 : ws * (ws - p_b) + 1, 0 : ws * (ws - p_b) + 1] = 1.0

        mask[W_s - 1 : (H_s - 1) * W_s : W_s, ...] = right
        mask[(H_s - 1) * W_s :, ...] = bottom
        mask[-1, ...] = bottom_right
        mask = mask.repeat(B, 1, 1)
    else:
        mask = None

    cls_tokens = cls_tokens.flatten(2).transpose(-2, -1)  # (N)x(H/sxK/s)xC
    cls_tokens = cls_tokens.reshape(-1, 1, cls_tokens.size(-1))  # (NxH/sxK/s)x1xC

    out = torch.cat((cls_tokens, patch_tokens), dim=1)

    return out, mask, p_l, p_r, p_t, p_b, B, C, H, W


def convert_to_spatial_layout(out, output_channels, B, H, W, kernel_size, mask, p_l, p_r, p_t, p_b):
    """
    Convert the token layer from flatten into 2-D, will be used to downsample the spatial dimension.
    """
    cls_tokens = out[:, 0:1, ...]
    patch_tokens = out[:, 1:, ...]
    # cls_tokens: (BxH/sxW/s)x(1)xC, patch_tokens: (BxH/sxW/s)x(ksxks)xC
    C = output_channels
    kernel_size = kernel_size[0]
    H_ks = H // kernel_size[0]
    W_ks = W // kernel_size[1]
    # reorganize data, need to convert back to cls_tokens: BxCxH/sxW/s, patch_tokens: BxCxHxW
    cls_tokens = cls_tokens.reshape(B, -1, C).transpose(-2, -1).reshape(B, C, H_ks, W_ks)
    patch_tokens = patch_tokens.transpose(1, 2).reshape((B, -1, kernel_size[0] * kernel_size[1] * C)).transpose(1, 2)
    patch_tokens = F.fold(patch_tokens, (H, W), kernel_size=kernel_size, stride=kernel_size, padding=(0, 0))

    if mask is not None:
        if p_b > 0:
            patch_tokens = patch_tokens[:, :, :-p_b, :]
        if p_r > 0:
            patch_tokens = patch_tokens[:, :, :, :-p_r]

    return cls_tokens, patch_tokens
