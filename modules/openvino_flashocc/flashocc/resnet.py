# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Adapted from mmdetection ResNet backbone interfaces (https://github.com/open-mmlab/mmdetection), Apache-2.0, Copyright (c) OpenMMLab.

import torch.nn as nn


class ResNet(nn.Module):
    def __init__(
        self,
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=-1,
        norm_cfg=None,
        norm_eval=False,
        with_cp=False,
        style="pytorch",
        pretrained=None,
        **kwargs,
    ):
        super().__init__()
        if depth != 50:
            raise ValueError(f"Standalone FlashOCC currently supports ResNet-50, got depth={depth}")
        try:
            from torchvision.models import resnet50
        except Exception as exc:
            raise ImportError("torchvision is required for the standalone FlashOCC ResNet backbone") from exc

        try:
            backbone = resnet50(weights=None)
        except TypeError:
            backbone = resnet50(pretrained=False)

        self.out_indices = tuple(out_indices)
        self.num_stages = num_stages
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.res_layers = ["layer1", "layer2", "layer3", "layer4"][:num_stages]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            x = getattr(self, layer_name)(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self
