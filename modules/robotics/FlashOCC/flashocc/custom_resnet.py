# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Adapted from BEVDet custom ResNet components (https://github.com/HuangJunJie2017/BEVDet), Apache-2.0, Copyright (c) OpenMMLab.

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from .layers import ConvModule


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, norm_cfg=None, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation,
                               dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, norm_cfg=None, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation,
                               dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class CustomResNet(nn.Module):
    def __init__(
        self,
        numC_input,
        num_layer=[2, 2, 2],
        num_channels=None,
        stride=[2, 2, 2],
        backbone_output_ids=None,
        norm_cfg=dict(type="BN"),
        with_cp=False,
        block_type="Basic",
    ):
        super().__init__()
        assert len(num_layer) == len(stride)
        num_channels = [numC_input * 2 ** (i + 1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) if backbone_output_ids is None else backbone_output_ids

        layers = []
        if block_type == "BottleNeck":
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [Bottleneck(inplanes=curr_numC, planes=num_channels[i] // 4, stride=stride[i],
                                    downsample=nn.Conv2d(curr_numC, num_channels[i], 3, stride[i], 1),
                                    norm_cfg=norm_cfg)]
                curr_numC = num_channels[i]
                layer.extend([Bottleneck(inplanes=curr_numC, planes=num_channels[i] // 4, stride=1,
                                         downsample=None, norm_cfg=norm_cfg) for _ in range(num_layer[i] - 1)])
                layers.append(nn.Sequential(*layer))
        elif block_type == "Basic":
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [BasicBlock(inplanes=curr_numC, planes=num_channels[i], stride=stride[i],
                                    downsample=nn.Conv2d(curr_numC, num_channels[i], 3, stride[i], 1),
                                    norm_cfg=norm_cfg)]
                curr_numC = num_channels[i]
                layer.extend([BasicBlock(inplanes=curr_numC, planes=num_channels[i], stride=1,
                                         downsample=None, norm_cfg=norm_cfg) for _ in range(num_layer[i] - 1)])
                layers.append(nn.Sequential(*layer))
        else:
            raise ValueError(f"Unsupported block_type: {block_type}")

        self.layers = nn.Sequential(*layers)
        self.with_cp = with_cp

    def forward(self, x):
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp and x_tmp.requires_grad:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats


class BasicBlock3D(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1, downsample=None):
        super().__init__()
        self.conv1 = ConvModule(channels_in, channels_out, kernel_size=3, stride=stride, padding=1,
                                bias=False, conv_cfg=dict(type="Conv3d"),
                                norm_cfg=dict(type="BN3d"), act_cfg=dict(type="ReLU", inplace=True))
        self.conv2 = ConvModule(channels_out, channels_out, kernel_size=3, stride=1, padding=1,
                                bias=False, conv_cfg=dict(type="Conv3d"),
                                norm_cfg=dict(type="BN3d"), act_cfg=None)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        x = self.conv1(x)
        x = self.conv2(x)
        return self.relu(x + identity)


class CustomResNet3D(nn.Module):
    def __init__(self, numC_input, num_layer=[2, 2, 2], num_channels=None,
                 stride=[2, 2, 2], backbone_output_ids=None, with_cp=False):
        super().__init__()
        assert len(num_layer) == len(stride)
        num_channels = [numC_input * 2 ** (i + 1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) if backbone_output_ids is None else backbone_output_ids
        layers = []
        curr_numC = numC_input
        for i in range(len(num_layer)):
            layer = [
                BasicBlock3D(
                    curr_numC,
                    num_channels[i],
                    stride=stride[i],
                    downsample=ConvModule(curr_numC, num_channels[i], kernel_size=3, stride=stride[i],
                                          padding=1, bias=False, conv_cfg=dict(type="Conv3d"),
                                          norm_cfg=dict(type="BN3d"), act_cfg=None),
                )
            ]
            curr_numC = num_channels[i]
            layer.extend([BasicBlock3D(curr_numC, curr_numC) for _ in range(num_layer[i] - 1)])
            layers.append(nn.Sequential(*layer))
        self.layers = nn.Sequential(*layers)
        self.with_cp = with_cp

    def forward(self, x):
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp and x_tmp.requires_grad:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats
