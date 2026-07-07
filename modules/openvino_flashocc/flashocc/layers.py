# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Reimplementation of FlashOCC under Apache-2.0-compatible terms.

import torch.nn as nn


class BaseModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


def _identity_decorator(fn=None, *args, **kwargs):
    if fn is not None and callable(fn):
        return fn

    def decorate(func):
        return func

    return decorate


force_fp32 = _identity_decorator
auto_fp16 = _identity_decorator


def build_conv_layer(cfg=None, *args, **kwargs):
    cfg = dict(cfg or {})
    layer_type = cfg.pop("type", "Conv2d")
    cfg.pop("im2col_step", None)
    if layer_type == "DCN":
        layer_type = "Conv2d"
    params = {**cfg, **kwargs}
    if layer_type in ("Conv", "Conv2d"):
        return nn.Conv2d(*args, **params)
    if layer_type == "Conv3d":
        return nn.Conv3d(*args, **params)
    raise ValueError(f"Unsupported convolution layer type: {layer_type}")


def build_norm_layer(cfg, num_features, postfix=""):
    cfg = dict(cfg or {"type": "BN"})
    layer_type = cfg.pop("type", "BN")
    requires_grad = cfg.pop("requires_grad", True)
    eps = cfg.pop("eps", 1e-5)
    momentum = cfg.pop("momentum", 0.1)

    if layer_type in ("BN", "BN2d", "BatchNorm2d"):
        name = f"bn{postfix}"
        layer = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, **cfg)
    elif layer_type in ("BN3d", "BatchNorm3d"):
        name = f"bn{postfix}"
        layer = nn.BatchNorm3d(num_features, eps=eps, momentum=momentum, **cfg)
    elif layer_type in ("LN", "LayerNorm"):
        name = f"ln{postfix}"
        layer = nn.LayerNorm(num_features, eps=eps, **cfg)
    elif layer_type in ("Identity", None):
        name = f"identity{postfix}"
        layer = nn.Identity()
    else:
        raise ValueError(f"Unsupported norm layer type: {layer_type}")

    for param in layer.parameters():
        param.requires_grad = requires_grad
    return name, layer


def build_activation_layer(cfg):
    if cfg is None:
        return None
    cfg = dict(cfg)
    layer_type = cfg.pop("type", "ReLU")
    if layer_type == "ReLU":
        return nn.ReLU(**cfg)
    if layer_type == "GELU":
        return nn.GELU(**cfg)
    if layer_type == "Sigmoid":
        return nn.Sigmoid()
    if layer_type == "Softplus":
        return nn.Softplus(**cfg)
    raise ValueError(f"Unsupported activation layer type: {layer_type}")


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias="auto",
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type="ReLU"),
        inplace=True,
        **kwargs,
    ):
        super().__init__()
        if bias == "auto":
            bias = norm_cfg is None
        if act_cfg is not None and "inplace" not in act_cfg:
            act_cfg = dict(act_cfg, inplace=inplace)

        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs,
        )
        self.with_norm = norm_cfg is not None
        if self.with_norm:
            norm_name, norm = build_norm_layer(norm_cfg, out_channels)
            self.norm_name = norm_name
            self.add_module(norm_name, norm)
        else:
            self.norm_name = None
        self.activate = build_activation_layer(act_cfg)

    @property
    def norm(self):
        return getattr(self, self.norm_name) if self.norm_name else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activate is not None:
            x = self.activate(x)
        return x
