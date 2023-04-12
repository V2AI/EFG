# Copyright IBM All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn


class Mlp1d(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv1d(in_features, hidden_features, kernel_size=1, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv1d(hidden_features, out_features, kernel_size=1, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Mlp2d(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super(LayerNorm2d, self).__init__()

        self.channels = channels
        self.eps = torch.tensor(eps)
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mean = input.mean(1, keepdim=True)
        std = torch.sqrt(input.var(1, unbiased=False, keepdim=True) + self.eps)
        out = (input - mean) / std
        if self.elementwise_affine:
            out = out * self.weight + self.bias
        return out

    def extra_repr(self):
        return "{channels}, eps={eps}, " "elementwise_affine={elementwise_affine}".format(**self.__dict__)


class LayerNorm1d(nn.Module):
    def __init__(self, channels, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super(LayerNorm1d, self).__init__()

        self.channels = channels
        self.eps = torch.tensor(eps)
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.zeros(1, channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, channels, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mean = input.mean(1, keepdim=True)
        std = torch.sqrt(input.var(1, unbiased=False, keepdim=True) + self.eps)
        out = (input - mean) / std
        if self.elementwise_affine:
            out = out * self.weight + self.bias
        return out

    def extra_repr(self):
        return "{channels}, eps={eps}, " "elementwise_affine={elementwise_affine}".format(**self.__dict__)


class Attention2d(nn.Module):
    def __init__(self, dim, out_dim=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()

        out_dim = dim if out_dim is None else out_dim
        self.num_heads = num_heads
        head_dim = out_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Conv2d(dim, out_dim * 3, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Conv2d(out_dim, out_dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.out_dim = out_dim

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).flatten(-2)  # BCHW -> B(3C)N
        qkv = qkv.reshape(B, 3, self.num_heads, self.out_dim // self.num_heads, H * W).permute(
            1, 0, 2, 4, 3
        )  # B(3C)N -> B3H(C/H)N -> 3BHN(C/H)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple), each BHN(C/H)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(-2, -1).reshape(B, self.out_dim, H, W)  # B H N (C / H) # B H (C/H) N

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
