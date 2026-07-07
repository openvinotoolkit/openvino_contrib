# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Reimplementation of FlashOCC under Apache-2.0-compatible terms.

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from .layers import ConvModule, build_norm_layer


class FPN_LSS(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factor=4,
        input_feature_index=(0, 2),
        norm_cfg=dict(type="BN"),
        extra_upsample=2,
        lateral=None,
        use_input_conv=False,
    ):
        super().__init__()
        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.out_channels = out_channels
        self.up = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)

        channels_factor = 2 if self.extra_upsample else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * channels_factor, kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * channels_factor, out_channels * channels_factor, kernel_size=3,
                      padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor)[1],
            nn.ReLU(inplace=True),
        )

        if self.extra_upsample:
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=extra_upsample, mode="bilinear", align_corners=True),
                nn.Conv2d(out_channels * channels_factor, out_channels, kernel_size=3, padding=1, bias=False),
                build_norm_layer(norm_cfg, out_channels)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            )

        self.lateral = lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(lateral, lateral, kernel_size=1, padding=0, bias=False),
                build_norm_layer(norm_cfg, lateral)[1],
                nn.ReLU(inplace=True),
            )

    def forward(self, feats):
        x2, x1 = feats[self.input_feature_index[0]], feats[self.input_feature_index[1]]
        if self.lateral:
            x2 = self.lateral_conv(x2)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        if self.extra_upsample:
            x = self.up2(x)
        return x


class LSSFPN3D(nn.Module):
    def __init__(self, in_channels, out_channels, with_cp=False):
        super().__init__()
        self.up1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True)
        self.conv = ConvModule(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                               bias=False, conv_cfg=dict(type="Conv3d"),
                               norm_cfg=dict(type="BN3d"), act_cfg=dict(type="ReLU", inplace=True))
        self.with_cp = with_cp

    def forward(self, feats):
        x_8, x_16, x_32 = feats
        x_16 = self.up1(x_16)
        x_32 = self.up2(x_32)
        x = torch.cat([x_8, x_16, x_32], dim=1)
        if self.with_cp and x.requires_grad:
            return checkpoint.checkpoint(self.conv, x)
        return self.conv(x)
