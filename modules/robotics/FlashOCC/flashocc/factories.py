# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

from .custom_resnet import CustomResNet
from .fpn import CustomFPN
from .lss_fpn import FPN_LSS
from .occ_head import BEVOCCHead2D
from .resnet import ResNet
from .view_transformer import LSSViewTransformer


BACKBONES = {
    "ResNet": ResNet,
    "CustomResNet": CustomResNet,
}

NECKS = {
    "CustomFPN": CustomFPN,
    "FPN_LSS": FPN_LSS,
    "LSSViewTransformer": LSSViewTransformer,
}

HEADS = {
    "BEVOCCHead2D": BEVOCCHead2D,
}


def _build(cfg, registry):
    cfg = deepcopy(cfg)
    layer_type = cfg.pop("type")
    if layer_type not in registry:
        raise KeyError(f"Unsupported standalone FlashOCC module type: {layer_type}")
    return registry[layer_type](**cfg)


def build_backbone(cfg):
    return _build(cfg, BACKBONES)


def build_neck(cfg):
    return _build(cfg, NECKS)


def build_head(cfg):
    return _build(cfg, HEADS)
