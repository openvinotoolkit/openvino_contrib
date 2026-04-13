#!/usr/bin/env python3
"""
Unified BEVFusion export script.

Exports ALL components to OpenVINO IR format:

  Neural Network Models (from PyTorch checkpoint):
    - camera_backbone_neck.xml/bin          (SwinT + FPN, batch=1)
    - camera_backbone_neck_b6.xml/bin       (SwinT + FPN, batch=6)
    - vtransform_dtransform.xml/bin         (depth transform, batch=1)
    - vtransform_dtransform_b6.xml/bin      (depth transform, batch=6)
    - vtransform_depthnet.xml/bin           (depth net, batch=1)
    - vtransform_depthnet_b6.xml/bin        (depth net, batch=6)
    - vtransform_downsample.xml/bin         (camera BEV downsample)
    - fuser.xml/bin                         (ConvFuser)
    - bev_decoder.xml/bin                   (SECOND + SECONDFPN)
    - transfusion_head.xml/bin              (TransFusion detection head)

  OpenVINO Custom Extension Models:
    - bev_pool.xml                          (no weights — stateless)
    - voxelize.xml                          (no weights — stateless)
    - sparse_encoder.xml/bin                (~10.8 MB packed weights)

  Auxiliary Files:
    - frustum.npy, vtransform_dx/bx/nx.npy
    - config.json
    - sparse_encoder_weights/               (raw NumPy weight files)

Usage:
    python export.py --checkpoint pretrained_models/bevfusion-det.pth \\
                     --output-dir openvino_model
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).parent


# ============================================================================
# Checkpoint Format Auto-Detection
# ============================================================================

def detect_checkpoint_format(state_dict):
    """Detect checkpoint format and return key prefix mappings."""
    keys = list(state_dict.keys())

    if any(k.startswith('encoders.camera.backbone') for k in keys):
        print("  Detected: MIT-HAN-Lab BEVFusion format")
        return {
            'format': 'mit-han-lab',
            'camera_backbone': 'encoders.camera.backbone.',
            'camera_neck': 'encoders.camera.neck.',
            'vtransform': 'encoders.camera.vtransform.',
            'lidar_encoder': 'encoders.lidar.backbone.',
            'fuser': 'fuser.',
            'decoder_backbone': 'decoder.backbone.',
            'decoder_neck': 'decoder.neck.',
            'head': 'heads.object.',
        }

    if any(k.startswith('img_backbone') for k in keys):
        print("  Detected: MMDetection3D format")
        return {
            'format': 'mmdet3d',
            'camera_backbone': 'img_backbone.',
            'camera_neck': 'img_neck.',
            'vtransform': 'view_transform.',
            'lidar_encoder': 'pts_middle_encoder.',
            'fuser': 'fusion_layer.',
            'decoder_backbone': 'pts_backbone.',
            'decoder_neck': 'pts_neck.',
            'head': 'bbox_head.',
        }

    raise ValueError(f"Unknown checkpoint format! First keys: {keys[:10]}")


# ============================================================================
# 1. Camera Backbone: SwinTransformer-Tiny
# ============================================================================

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        rpb = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        rpb = rpb.permute(2, 0, 1).contiguous()
        attn = attn + rpb.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            nH, nW = Hp // self.window_size, Wp // self.window_size
            mask_windows = img_mask.view(1, nH, self.window_size, nW, self.window_size, 1)
            mask_windows = mask_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        else:
            shifted_x = x
            attn_mask = None

        nH, nW = Hp // self.window_size, Wp // self.window_size
        windows = shifted_x.view(B, nH, self.window_size, nW, self.window_size, C)
        windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(
            -1, self.window_size * self.window_size, C)
        attn_windows = self.attn(windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = attn_windows.view(B, nH, nW, self.window_size, self.window_size, C)
        shifted_x = shifted_x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.ffn(self.norm2(x))
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.reduction = nn.Linear(4 * in_channels, 2 * in_channels, bias=False)
        self.norm = nn.LayerNorm(4 * in_channels)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        _, new_H, new_W, _ = x.shape
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, new_H, new_W


class SwinStage(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=7, downsample=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(SwinBlock(
                dim, num_heads, window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2))
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        x_before_ds = x
        H_before_ds, W_before_ds = H, W
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        return x, H, W, x_before_ds, H_before_ds, W_before_ds


class SwinTransformerBackbone(nn.Module):
    def __init__(self, embed_dims=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, out_indices=[1, 2, 3]):
        super().__init__()
        self.out_indices = out_indices
        self.patch_embed = nn.ModuleDict({
            'projection': nn.Conv2d(3, embed_dims, 4, stride=4, bias=True),
            'norm': nn.LayerNorm(embed_dims),
        })
        dims = [embed_dims * (2 ** i) for i in range(len(depths))]
        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            self.stages.append(SwinStage(
                dims[i], depths[i], num_heads[i], window_size,
                downsample=(i < len(depths) - 1)))
        self.norms = nn.ModuleDict()
        for i in out_indices:
            self.norms[str(i)] = nn.LayerNorm(dims[i])

    def forward(self, x):
        x = self.patch_embed['projection'](x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.patch_embed['norm'](x)
        outs = []
        for i, stage in enumerate(self.stages):
            x, H, W, x_before_ds, H_bds, W_bds = stage(x, H, W)
            if i in self.out_indices:
                norm = self.norms[str(i)]
                out = norm(x_before_ds)
                out = out.view(B, H_bds, W_bds, -1).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return outs


# ============================================================================
# 2. Camera Neck: GeneralizedLSSFPN
# ============================================================================

class GeneralizedLSSFPN(nn.Module):
    def __init__(self, in_channels=[192, 384, 768], out_channels=256):
        super().__init__()
        lat1_in = in_channels[1] + in_channels[2]
        lat0_in = in_channels[0] + out_channels
        self.lateral_convs = nn.ModuleList([
            nn.ModuleDict({
                'conv': nn.Conv2d(lat0_in, out_channels, 1, bias=False),
                'bn': nn.BatchNorm2d(out_channels),
            }),
            nn.ModuleDict({
                'conv': nn.Conv2d(lat1_in, out_channels, 1, bias=False),
                'bn': nn.BatchNorm2d(out_channels),
            }),
        ])
        self.fpn_convs = nn.ModuleList([
            nn.ModuleDict({
                'conv': nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                'bn': nn.BatchNorm2d(out_channels),
            }),
            nn.ModuleDict({
                'conv': nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                'bn': nn.BatchNorm2d(out_channels),
            }),
        ])

    def forward(self, inputs):
        assert len(inputs) == 3
        up3 = F.interpolate(inputs[2], size=inputs[1].shape[2:], mode='bilinear', align_corners=False)
        cat1 = torch.cat([inputs[1], up3], dim=1)
        lat1 = F.relu(self.lateral_convs[1]['bn'](self.lateral_convs[1]['conv'](cat1)))
        up1 = F.interpolate(lat1, size=inputs[0].shape[2:], mode='bilinear', align_corners=False)
        cat0 = torch.cat([inputs[0], up1], dim=1)
        lat0 = F.relu(self.lateral_convs[0]['bn'](self.lateral_convs[0]['conv'](cat0)))
        out0 = F.relu(self.fpn_convs[0]['bn'](self.fpn_convs[0]['conv'](lat0)))
        out1 = F.relu(self.fpn_convs[1]['bn'](self.fpn_convs[1]['conv'](lat1)))
        return [out0, out1]


class CameraBackboneNeck(nn.Module):
    """Combined SwinT backbone + GeneralizedLSSFPN neck.
    Input: [B, 3, 256, 704] -> Output: [B, 256, 32, 88]
    """
    def __init__(self):
        super().__init__()
        self.backbone = SwinTransformerBackbone(
            embed_dims=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
            window_size=7, out_indices=[1, 2, 3])
        self.neck = GeneralizedLSSFPN(
            in_channels=[192, 384, 768], out_channels=256)

    def forward(self, x):
        feats = self.backbone(x)
        neck_out = self.neck(feats)
        return neck_out[0]


# ============================================================================
# 1b. Alternative Camera Backbone: ResNet50
# ============================================================================

class Bottleneck(nn.Module):
    """ResNet bottleneck block with 1x1 -> 3x3 -> 1x1 convolutions."""
    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        super().__init__()
        out_channels = mid_channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.relu(out)


class ResNet50Backbone(nn.Module):
    """ResNet-50 backbone outputting multi-scale features [512, 1024, 2048]."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)

    def _make_layer(self, in_channels, mid_channels, num_blocks, stride):
        out_channels = mid_channels * Bottleneck.expansion
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [Bottleneck(in_channels, mid_channels, stride, downsample)]
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(out_channels, mid_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c3, c4, c5]


class CameraBackboneNeckResNet(nn.Module):
    """Combined ResNet50 backbone + GeneralizedLSSFPN neck.
    Input: [B, 3, 256, 704] -> Output: [B, 256, 32, 88]
    Uses ImageNet-pretrained ResNet50 weights when available.
    """
    def __init__(self):
        super().__init__()
        self.backbone = ResNet50Backbone()
        self.neck = GeneralizedLSSFPN(
            in_channels=[512, 1024, 2048], out_channels=256)
        self._init_pretrained()

    def _init_pretrained(self):
        try:
            from torchvision.models import resnet50, ResNet50_Weights
            pretrained = resnet50(weights=ResNet50_Weights.DEFAULT)
            src_sd = pretrained.state_dict()
            tgt_sd = self.backbone.state_dict()
            loaded = 0
            for k, v in src_sd.items():
                if k in tgt_sd and tgt_sd[k].shape == v.shape:
                    tgt_sd[k] = v
                    loaded += 1
            self.backbone.load_state_dict(tgt_sd)
            print(f"  ResNet50 backbone: loaded {loaded} pretrained params from torchvision")
        except Exception as e:
            print(f"  ResNet50 backbone: random init (pretrained load failed: {e})")

    def forward(self, x):
        feats = self.backbone(x)
        neck_out = self.neck(feats)
        return neck_out[0]


def load_camera_backbone_neck_weights(model, state_dict, prefixes):
    """Load weights from BEVFusion checkpoint into standalone model."""
    prefix_bb = prefixes['camera_backbone']
    bb_state = {}
    for k, v in state_dict.items():
        if k.startswith(prefix_bb):
            new_key = k[len(prefix_bb):]
            new_key = new_key.replace('attn.w_msa.', 'attn.')
            new_key = new_key.replace('ffn.layers.0.0.', 'ffn.0.')
            new_key = new_key.replace('ffn.layers.1.', 'ffn.2.')
            for ni in [1, 2, 3]:
                if new_key.startswith(f'norm{ni}.'):
                    new_key = new_key.replace(f'norm{ni}.', f'norms.{ni}.')
            bb_state['backbone.' + new_key] = v

    prefix_neck = prefixes['camera_neck']
    neck_state = {}
    for k, v in state_dict.items():
        if k.startswith(prefix_neck):
            new_key = k[len(prefix_neck):]
            neck_state['neck.' + new_key] = v

    combined = {**bb_state, **neck_state}
    missing, unexpected = model.load_state_dict(combined, strict=False)
    loaded = len(combined) - len(missing)
    print(f"  Camera backbone+neck: loaded {loaded}/{len(combined)} params, "
          f"missing {len(missing)}, unexpected {len(unexpected)}")
    return len(missing) == 0


# ============================================================================
# 3. VTransform (DepthLSSTransform) components
# ============================================================================

def build_vtransform(state_dict, prefixes):
    """Build DepthLSSTransform components from checkpoint."""
    dtransform = nn.Sequential(
        nn.Conv2d(1, 8, 1),
        nn.BatchNorm2d(8),
        nn.ReLU(True),
        nn.Conv2d(8, 32, 5, stride=4, padding=2),
        nn.BatchNorm2d(32),
        nn.ReLU(True),
        nn.Conv2d(32, 64, 5, stride=2, padding=2),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
    )

    depthnet = nn.Sequential(
        nn.Conv2d(320, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(True),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(True),
        nn.Conv2d(256, 198, 1),
    )

    downsample = nn.Sequential(
        nn.Conv2d(80, 80, 3, padding=1, bias=False),
        nn.BatchNorm2d(80),
        nn.ReLU(True),
        nn.Conv2d(80, 80, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(80),
        nn.ReLU(True),
        nn.Conv2d(80, 80, 3, padding=1, bias=False),
        nn.BatchNorm2d(80),
        nn.ReLU(True),
    )

    prefix = prefixes['vtransform']
    for module, name in [(dtransform, 'dtransform'), (depthnet, 'depthnet'),
                         (downsample, 'downsample')]:
        mod_state = {}
        for k, v in state_dict.items():
            if k.startswith(prefix + name + '.'):
                mod_state[k[len(prefix + name + '.'):]] = v
        missing, unexpected = module.load_state_dict(mod_state)
        module.eval()
        print(f"  {name}: loaded {len(mod_state)} params, missing={len(missing)}")

    frustum = state_dict[prefix + 'frustum']
    dx = state_dict.get(prefix + 'dx', torch.tensor([0.6, 0.6, 20.0]))
    bx = state_dict.get(prefix + 'bx', torch.tensor([-53.7, -53.7, 0.0]))
    nx = state_dict.get(prefix + 'nx', torch.tensor([180, 180, 1]))

    return dtransform, depthnet, downsample, frustum, dx, bx, nx


# ============================================================================
# 4. Fuser (ConvFuser)
# ============================================================================

def build_fuser(state_dict, prefixes):
    """Build fuser: Conv2d(336->256, 3, p=1) + BN + ReLU."""
    fuser = nn.Sequential(
        nn.Conv2d(336, 256, 3, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(True),
    )
    prefix = prefixes['fuser']
    fuser_state = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            new_key = new_key.replace('conv.', '')
            fuser_state[new_key] = v
    fuser.load_state_dict(fuser_state)
    fuser.eval()
    print(f"  Fuser: loaded {len(fuser_state)} params")
    return fuser


# ============================================================================
# 5. BEV Decoder (SECOND backbone + SECONDFPN neck)
# ============================================================================

def build_decoder(state_dict, prefixes):
    """Build SECOND backbone + SECONDFPN neck from checkpoint weights."""
    bb_prefix = prefixes['decoder_backbone']
    neck_prefix = prefixes['decoder_neck']

    class SECOND(nn.Module):
        def __init__(self, sd, prefix):
            super().__init__()
            self.blocks = nn.ModuleList()
            for bi in range(2):
                bp = f'{prefix}blocks.{bi}.'
                layers = []
                for li in range(6):
                    base = li * 3
                    w = sd[f'{bp}{base}.weight']
                    s = 2 if (bi == 1 and li == 0) else 1
                    conv = nn.Conv2d(w.shape[1], w.shape[0], w.shape[2],
                                     stride=s, padding=w.shape[2] // 2, bias=False)
                    conv.weight.data = w
                    layers.append(conv)
                    bn = nn.BatchNorm2d(w.shape[0], eps=1e-3, momentum=0.01)
                    bn.weight.data = sd[f'{bp}{base + 1}.weight']
                    bn.bias.data = sd[f'{bp}{base + 1}.bias']
                    bn.running_mean.data = sd[f'{bp}{base + 1}.running_mean']
                    bn.running_var.data = sd[f'{bp}{base + 1}.running_var']
                    layers.append(bn)
                    layers.append(nn.ReLU(inplace=True))
                self.blocks.append(nn.Sequential(*layers))

        def forward(self, x):
            outs = []
            for block in self.blocks:
                x = block(x)
                outs.append(x)
            return outs

    class SECONDFPN(nn.Module):
        def __init__(self, sd, prefix):
            super().__init__()
            self.deblocks = nn.ModuleList()
            p = prefix
            w0 = sd[f'{p}deblocks.0.0.weight']
            conv0 = nn.Conv2d(w0.shape[1], w0.shape[0], w0.shape[2], stride=1, bias=False)
            conv0.weight.data = w0
            bn0 = nn.BatchNorm2d(w0.shape[0], eps=1e-3, momentum=0.01)
            bn0.weight.data = sd[f'{p}deblocks.0.1.weight']
            bn0.bias.data = sd[f'{p}deblocks.0.1.bias']
            bn0.running_mean.data = sd[f'{p}deblocks.0.1.running_mean']
            bn0.running_var.data = sd[f'{p}deblocks.0.1.running_var']
            self.deblocks.append(nn.Sequential(conv0, bn0, nn.ReLU(inplace=True)))

            w1 = sd[f'{p}deblocks.1.0.weight']
            deconv1 = nn.ConvTranspose2d(w1.shape[0], w1.shape[1], w1.shape[2],
                                         stride=2, bias=False)
            deconv1.weight.data = w1
            bn1 = nn.BatchNorm2d(w1.shape[1], eps=1e-3, momentum=0.01)
            bn1.weight.data = sd[f'{p}deblocks.1.1.weight']
            bn1.bias.data = sd[f'{p}deblocks.1.1.bias']
            bn1.running_mean.data = sd[f'{p}deblocks.1.1.running_mean']
            bn1.running_var.data = sd[f'{p}deblocks.1.1.running_var']
            self.deblocks.append(nn.Sequential(deconv1, bn1, nn.ReLU(inplace=True)))

        def forward(self, outs):
            ups = [block(outs[i]) for i, block in enumerate(self.deblocks)]
            return torch.cat(ups, dim=1)

    backbone = SECOND(state_dict, bb_prefix)
    neck = SECONDFPN(state_dict, neck_prefix)
    backbone.eval()
    neck.eval()
    print(f"  Decoder: built SECOND backbone + SECONDFPN neck")
    return backbone, neck


# ============================================================================
# 6. TransFusion Detection Head
# ============================================================================

def build_transfusion_head(state_dict, prefixes):
    """Build TransFusion detection head from checkpoint weights."""

    class PositionEmbedding(nn.Module):
        def __init__(self, hc=128):
            super().__init__()
            self.position_embedding_head = nn.Sequential(
                nn.Conv1d(2, hc, 1), nn.BatchNorm1d(hc), nn.ReLU(True),
                nn.Conv1d(hc, hc, 1))
        def forward(self, pos):
            return self.position_embedding_head(pos)

    class TransformerDecoderLayer(nn.Module):
        def __init__(self, hc=128, nh=8, fc=256, do=0.0):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(hc, nh, dropout=do, batch_first=False)
            self.multihead_attn = nn.MultiheadAttention(hc, nh, dropout=do, batch_first=False)
            self.linear1 = nn.Linear(hc, fc)
            self.linear2 = nn.Linear(fc, hc)
            self.norm1 = nn.LayerNorm(hc)
            self.norm2 = nn.LayerNorm(hc)
            self.norm3 = nn.LayerNorm(hc)
            self.self_posembed = PositionEmbedding(hc)
            self.cross_posembed = PositionEmbedding(hc)

        def forward(self, query, key, query_pos, key_pos):
            qpe = self.self_posembed(query_pos)
            kpe = self.cross_posembed(key_pos)
            q = (query + qpe).permute(2, 0, 1)
            v_s = query.permute(2, 0, 1)
            q2, _ = self.self_attn(q, q, v_s)
            query2 = query + q2.permute(1, 2, 0)
            query2 = self.norm1(query2.permute(0, 2, 1)).permute(0, 2, 1)
            q = (query2 + qpe).permute(2, 0, 1)
            k = (key + kpe).permute(2, 0, 1)
            v = key.permute(2, 0, 1)
            q2, _ = self.multihead_attn(q, k, v)
            query2 = query2 + q2.permute(1, 2, 0)
            query2 = self.norm2(query2.permute(0, 2, 1)).permute(0, 2, 1)
            q = query2.permute(0, 2, 1)
            q2 = self.linear2(F.relu(self.linear1(q)))
            query3 = q + q2
            query3 = self.norm3(query3).permute(0, 2, 1)
            return query3

    class ConvBNBlock(nn.Module):
        def __init__(self, ic, oc):
            super().__init__()
            self.conv = nn.Conv1d(ic, oc, 1, bias=False)
            self.bn = nn.BatchNorm1d(oc)
            self.relu = nn.ReLU(True)
        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class SeparateHead(nn.Module):
        def __init__(self, hc=128, nc=10):
            super().__init__()
            self.center = nn.ModuleList([ConvBNBlock(hc, 64), nn.Conv1d(64, 2, 1)])
            self.height = nn.ModuleList([ConvBNBlock(hc, 64), nn.Conv1d(64, 1, 1)])
            self.dim = nn.ModuleList([ConvBNBlock(hc, 64), nn.Conv1d(64, 3, 1)])
            self.rot = nn.ModuleList([ConvBNBlock(hc, 64), nn.Conv1d(64, 2, 1)])
            self.vel = nn.ModuleList([ConvBNBlock(hc, 64), nn.Conv1d(64, 2, 1)])
            self.heatmap = nn.ModuleList([ConvBNBlock(hc, 64), nn.Conv1d(64, nc, 1)])
        def forward(self, x):
            return {n: getattr(self, n)[1](getattr(self, n)[0](x))
                    for n in ['center', 'height', 'dim', 'rot', 'vel', 'heatmap']}

    class TransFusionHead(nn.Module):
        def __init__(self):
            super().__init__()
            ic, hc, nc, np_, nks = 512, 128, 10, 200, 3
            self.hc = hc
            self.nc = nc
            self.np_ = np_
            self.nks = nks
            self.shared_conv = nn.Conv2d(ic, hc, 3, padding=1)
            self.heatmap_head = nn.ModuleList([
                nn.ModuleDict({
                    'conv': nn.Conv2d(hc, hc, 3, padding=1, bias=False),
                    'bn': nn.BatchNorm2d(hc),
                }),
                nn.Conv2d(hc, nc, 3, padding=1),
            ])
            self.class_encoding = nn.Conv1d(nc, hc, 1)
            self.decoder = nn.ModuleList([TransformerDecoderLayer()])
            self.prediction_heads = nn.ModuleList([SeparateHead()])
            xs, ys = 180, 180
            batch_x, batch_y = torch.meshgrid(
                torch.linspace(0, xs - 1, xs),
                torch.linspace(0, ys - 1, ys))
            batch_x = batch_x + 0.5
            batch_y = batch_y + 0.5
            coord = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
            self.register_buffer('bev_pos', coord.view(1, 2, -1).permute(0, 2, 1))

        def forward(self, bev_features):
            B = bev_features.shape[0]
            fusion_feat = self.shared_conv(bev_features)
            x = self.heatmap_head[0]['conv'](fusion_feat)
            x = self.heatmap_head[0]['bn'](x)
            x = F.relu(x)
            dense_heatmap = self.heatmap_head[1](x)
            heatmap = dense_heatmap.detach().sigmoid()
            pad = self.nks // 2
            local_max = torch.zeros_like(heatmap)
            local_max_inner = F.max_pool2d(heatmap, kernel_size=self.nks, stride=1,
                                           padding=0)
            local_max[:, :, pad:(-pad), pad:(-pad)] = local_max_inner
            local_max[:, 8] = F.max_pool2d(heatmap[:, 8:9], kernel_size=1, stride=1,
                                           padding=0)[:, 0]
            local_max[:, 9] = F.max_pool2d(heatmap[:, 9:10], kernel_size=1, stride=1,
                                           padding=0)[:, 0]
            heatmap = heatmap * (heatmap == local_max)
            feat_flat = fusion_feat.view(B, self.hc, -1)
            hm_flat = heatmap.view(B, self.nc, -1)
            hm_all = hm_flat.view(B, -1)
            _, top = torch.topk(hm_all, k=self.np_, dim=-1, largest=True, sorted=True)
            top_class = top // hm_flat.shape[-1]
            top_idx = top % hm_flat.shape[-1]
            qf = feat_flat.gather(
                index=top_idx[:, None, :].expand(-1, self.hc, -1), dim=-1)
            oh = F.one_hot(top_class, num_classes=self.nc).permute(0, 2, 1).float()
            qf = qf + self.class_encoding(oh)
            bp = self.bev_pos.expand(B, -1, -1)
            qp = bp.gather(
                index=top_idx[:, :, None].expand(-1, -1, 2), dim=1).permute(0, 2, 1)
            key = feat_flat
            kp = bp.permute(0, 2, 1)
            for i in range(1):
                qf = self.decoder[i](qf, key, qp, kp)
                res = self.prediction_heads[i](qf)
                res['center'] = res['center'] + qp
                qp = res['center'].detach().clone()
            query_heatmap_score = hm_flat.gather(
                index=top_idx[:, None, :].expand(-1, self.nc, -1), dim=-1)
            res['query_heatmap_score'] = query_heatmap_score
            res['dense_heatmap'] = dense_heatmap
            res['top_proposals_index'] = top_idx
            res['top_proposals_class'] = top_class
            return res

    head = TransFusionHead()
    model_dict = head.state_dict()
    prefix = prefixes['head']
    loaded = 0
    for k, v in state_dict.items():
        if k.startswith(prefix):
            mk = k[len(prefix):]
            if mk in model_dict and model_dict[mk].shape == v.shape:
                model_dict[mk] = v
                loaded += 1
    head.load_state_dict(model_dict)
    head.eval()
    print(f"  TransFusion head: loaded {loaded} weights")
    return head


# ============================================================================
# Export Utilities
# ============================================================================

D_BINS = 118  # depth bins
C_FEAT = 80   # context feature channels


class CameraEncoderMerged(nn.Module):
    """Merged: backbone+neck + dtransform + depthnet → single model.

    Eliminates 2 model call overheads (3 calls → 1).
    Input:  images [B,3,256,704], depth_maps [B,1,256,704]
    Output: depth_logits [B,118,32,88], context_feats [B,80,32,88]
    """
    def __init__(self, backbone_neck, dtransform, depthnet):
        super().__init__()
        self.backbone_neck = backbone_neck
        self.dtransform = dtransform
        self.depthnet = depthnet

    def forward(self, images, depth_maps):
        cam_feats = self.backbone_neck(images)       # [B,256,32,88]
        depth_feats = self.dtransform(depth_maps)     # [B,64,32,88]
        combined = torch.cat([cam_feats, depth_feats], dim=1)  # [B,320,32,88]
        daf = self.depthnet(combined)                 # [B,198,32,88]
        depth_logits = daf[:, :D_BINS]                # [B,118,32,88]
        context_feats = daf[:, D_BINS:]               # [B,80,32,88]
        return depth_logits, context_feats


class DetectionHeadMerged(nn.Module):
    """Merged: downsample + fuser + bev_decoder → single model.

    Eliminates 2 model call overheads (3 calls → 1).
    TransFusion head is kept separate (complex topology).
    Input:  camera_bev [1,80,360,360], lidar_bev [1,256,180,180]
    Output: decoder features [1,512,180,180]
    """
    def __init__(self, downsample, fuser, decoder_backbone, decoder_neck):
        super().__init__()
        self.downsample = downsample
        self.fuser = fuser
        self.decoder_backbone = decoder_backbone
        self.decoder_neck = decoder_neck

    def forward(self, camera_bev, lidar_bev):
        cam_ds = self.downsample(camera_bev)                    # [1,80,180,180]
        fused = torch.cat([cam_ds, lidar_bev], dim=1)           # [1,336,180,180]
        fused = self.fuser(fused)                                # [1,256,180,180]
        dec_outs = self.decoder_backbone(fused)                  # list of tensors
        decoded = self.decoder_neck(dec_outs)                    # [1,512,180,180]
        return decoded


def export_to_openvino(model, dummy_input, name, input_names, output_names,
                       output_dir, use_direct=False):
    """Export a PyTorch model to OpenVINO IR via ONNX or direct conversion."""
    import openvino as ov

    onnx_path = output_dir / f'{name}.onnx'
    xml_path = output_dir / f'{name}.xml'

    with torch.no_grad():
        if isinstance(dummy_input, tuple):
            out = model(*dummy_input)
        else:
            out = model(dummy_input)
        if isinstance(out, dict):
            for k, v in out.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: {v.shape}")
        elif isinstance(out, (tuple, list)):
            for i, v in enumerate(out):
                print(f"    output_{i}: {v.shape}")
        else:
            print(f"    output: {out.shape}")

    if use_direct:
        # Direct PyTorch → OpenVINO (skip ONNX)
        try:
            # Build static input spec from dummy_input
            if isinstance(dummy_input, tuple):
                input_spec = [ov.PartialShape(list(t.shape)) for t in dummy_input]
            else:
                input_spec = [ov.PartialShape(list(dummy_input.shape))]
            ov_model = ov.convert_model(model, example_input=dummy_input,
                                        input=input_spec)
            # Rename inputs
            for i, inp_name in enumerate(input_names):
                ov_model.inputs[i].get_tensor().set_names({inp_name})
            # Rename outputs
            for i, out_name in enumerate(output_names):
                ov_model.outputs[i].get_tensor().set_names({out_name})
            ov.save_model(ov_model, str(xml_path))
            print(f"  Saved (direct): {xml_path}")
            return xml_path
        except Exception as e:
            print(f"  Direct conversion failed ({e}), falling back to ONNX")

    torch.onnx.export(
        model, dummy_input, str(onnx_path),
        input_names=input_names, output_names=output_names,
        opset_version=17, do_constant_folding=True)

    ov_model = ov.convert_model(str(onnx_path))
    ov.save_model(ov_model, str(xml_path))
    print(f"  Saved: {xml_path}")
    return xml_path


# ============================================================================
# Extension Model Export (BEV Pool, Voxelize, Sparse Encoder)
# ============================================================================

def export_extension_models(output_dir: Path, sparse_weights_dir: Path):
    """Export the custom OpenVINO extension IR models.

    These are not neural networks — they are custom ops executed by the
    .so extension libraries alongside the neural network models.
    """
    import openvino as ov

    core = ov.Core()

    # Load extension libraries (needed for validation)
    ext_root = PROJECT_DIR / 'openvino_extensions'
    for name, lib in [
        ('BEVPool', ext_root / 'bev_pool' / 'build' / 'libopenvino_bevpool_extension.so'),
        ('Voxelize', ext_root / 'voxelize' / 'build' / 'libopenvino_voxelize_extension.so'),
        ('SparseEncoder', ext_root / 'sparse_encoder' / 'build' / 'libopenvino_sparse_encoder_extension.so'),
    ]:
        if lib.exists():
            core.add_extension(str(lib))
            print(f"    Loaded {name} extension")
        else:
            print(f"    Warning: {name} extension not built: {lib}")

    # 1. BEV Pool
    from bevfusion.bev_pool_extension import BEVPoolExtension, HAS_BEVPOOL_EXTENSION
    if HAS_BEVPOOL_EXTENSION:
        ext = BEVPoolExtension(core, 'GPU')
        ext.export_model(output_dir)
    else:
        print("    Warning: BEVPool extension not available, skipping")

    # 2. Voxelize
    from bevfusion.voxelize_extension import VoxelizeExtension, HAS_VOXELIZE_EXTENSION
    if HAS_VOXELIZE_EXTENSION:
        ext = VoxelizeExtension(core, 'GPU')
        ext.export_model(output_dir)
    else:
        print("    Warning: Voxelize extension not available, skipping")

    # 3. Sparse Encoder
    from bevfusion.sparse_encoder_extension import (
        SparseEncoderExtension, HAS_SPARSE_ENCODER_EXTENSION,
    )
    if HAS_SPARSE_ENCODER_EXTENSION:
        ext = SparseEncoderExtension(core, str(sparse_weights_dir))
        ext.export_model(output_dir)
    else:
        print("    Warning: Sparse Encoder extension not available, skipping")


# ============================================================================
# Post-Export Optimization: INT8 Quantization (camera_encoder_b6 only)
# ============================================================================

def quantize_camera_encoder_int8(output_dir: Path, data_pkl: str,
                                  data_root: str, num_cal_samples: int = 30):
    """Quantize camera_encoder_b6 to INT8 using NNCF with real calibration data.

    NOTE: Only camera_encoder_b6 is quantized. full_detection INT8 with
    FakeQuantize activation nodes causes CL_OUT_OF_RESOURCES on Panther Lake
    iGPU and must NOT be quantized.
    """
    import openvino as ov

    xml_path = output_dir / 'camera_encoder_b6.xml'
    int8_path = output_dir / 'camera_encoder_b6_int8.xml'

    if not xml_path.exists():
        print("  Skipping: camera_encoder_b6.xml not found")
        return False

    # Check calibration data availability
    if not Path(data_pkl).exists():
        print(f"  Skipping INT8: calibration data not found at {data_pkl}")
        print("  Run with --data-pkl and --data-root to enable quantization")
        return False

    try:
        import nncf
    except ImportError:
        print("  Skipping INT8: nncf package not installed (pip install nncf)")
        return False

    print(f"\n  Collecting calibration data ({num_cal_samples} samples)...")

    # Load dataset info
    with open(data_pkl, 'rb') as f:
        data = pickle.load(f)
    infos = data['infos'] if isinstance(data, dict) else data
    data_root = Path(data_root)

    # Import pipeline helpers for data loading
    from run_inference_standalone import (
        load_lidar_points, load_camera_images, create_depth_maps,
        get_camera_transforms,
    )

    cal_data = []
    n = min(num_cal_samples, len(infos))
    for i in range(n):
        info = infos[i]
        points = load_lidar_points(info, data_root)
        images, aug_mats = load_camera_images(info, data_root)
        l2i, c2l, intr = get_camera_transforms(info)
        depth_maps = create_depth_maps(points, l2i, aug_mats)
        cal_data.append({
            'images': np.ascontiguousarray(images),
            'depth_maps': np.ascontiguousarray(depth_maps),
        })
        if (i + 1) % 10 == 0:
            print(f"    Collected {i + 1}/{n} samples")
    print(f"    Collected {n} calibration samples total")

    # Quantize
    print(f"  Quantizing camera_encoder_b6 -> INT8...")
    core = ov.Core()
    model = core.read_model(str(xml_path))

    def transform_fn(data_item):
        inputs = {}
        for inp in model.inputs:
            name = inp.get_any_name()
            if name in data_item:
                inputs[name] = data_item[name]
        return inputs

    cal_dataset = nncf.Dataset(cal_data, transform_fn)

    quantized = nncf.quantize(
        model, cal_dataset,
        preset=nncf.QuantizationPreset.MIXED,
        advanced_parameters=nncf.AdvancedQuantizationParameters(
            overflow_fix=nncf.OverflowFix.DISABLE,
        ),
    )

    ov.save_model(quantized, str(int8_path))

    orig_size = os.path.getsize(str(xml_path).replace('.xml', '.bin'))
    new_size = os.path.getsize(str(int8_path).replace('.xml', '.bin'))
    print(f"  INT8 saved: {int8_path}")
    print(f"  Size: {orig_size / 1e6:.1f}MB -> {new_size / 1e6:.1f}MB "
          f"({new_size / orig_size * 100:.0f}%)")
    return True


# ============================================================================
# Post-Export Optimization: Per-Class TopK (full_detection)
# ============================================================================

def optimize_full_detection_topk(output_dir: Path):
    """Replace global TopK-200 on [1,324000] with per-class TopK-20 on [10,32400].

    The original full_detection model selects the top-200 proposals from ALL
    10 classes x 32400 spatial positions flattened into [1,324000]. This TopK
    takes ~28ms on Panther Lake iGPU because GPU TopK scales with input size.

    This optimization reshapes to [10,32400] and does TopK k=20 per class,
    reducing TopK from 28ms to ~3ms. The class indices and spatial indices are
    reconstructed from fixed constants, eliminating the expensive Floor/FloorMod
    integer division nodes.

    The optimized model overwrites full_detection.xml. A backup is saved as
    full_detection_original.xml.
    """
    import openvino as ov
    from openvino.runtime import opset13 as opset

    xml_path = output_dir / 'full_detection.xml'
    if not xml_path.exists():
        print("  Skipping: full_detection.xml not found")
        return False

    print("  Applying per-class TopK optimization to full_detection...")

    core = ov.Core()
    model = core.read_model(str(xml_path))

    # ── Find the relevant nodes ──────────────────────────────────────
    nodes = {}
    for op in model.get_ordered_ops():
        name = op.get_friendly_name()
        op_type = op.get_type_name()

        # The Reshape that creates [1, 324000] from [1, 10, 32400]
        if op_type == 'Reshape':
            out_shape = op.output(0).get_partial_shape()
            if len(out_shape) == 2:
                dims = [d.get_length() if d.is_static else -1 for d in out_shape]
                if dims == [1, 324000]:
                    nodes['reshape_flat'] = op

        # TopK node
        if op_type == 'TopK':
            nodes['topk'] = op

        # The Reshape that creates [1, 10, 32400] from [1, 10, 180, 180]
        if op_type == 'Reshape':
            out_shape = op.output(0).get_partial_shape()
            if len(out_shape) == 3:
                dims = [d.get_length() if d.is_static else -1 for d in out_shape]
                if dims == [1, 10, 32400]:
                    nodes['reshape_3d'] = op

        # Floor (class_idx = top // 32400)
        if op_type == 'Floor':
            nodes['floor'] = op

        # FloorMod (spatial_idx = top % 32400)
        if op_type == 'FloorMod':
            nodes['floormod'] = op

    required = ['reshape_3d', 'reshape_flat', 'topk', 'floor', 'floormod']
    missing = [k for k in required if k not in nodes]
    if missing:
        print(f"  Skipping: could not find required nodes: {missing}")
        print("  (Model structure may have changed)")
        return False

    # ── Build per-class TopK replacement ─────────────────────────────
    # Reshape [1, 10, 32400] -> [10, 32400]
    new_reshape = opset.reshape(
        nodes['reshape_3d'].output(0),
        opset.constant(np.array([10, 32400], dtype=np.int64)),
        special_zero=False,
    )

    # TopK k=20 per class on [10, 32400] -> values [10, 20], indices [10, 20]
    k20 = opset.constant(np.array(20, dtype=np.int64))
    new_topk = opset.topk(new_reshape, k20, axis=-1, mode='max', sort='value')

    # Reshape indices [10, 20] -> [1, 200] to match downstream expectations
    spatial_indices = opset.reshape(
        new_topk.output(1),
        opset.constant(np.array([1, 200], dtype=np.int64)),
        special_zero=False,
    )

    # Build constant class indices: [0,0,...(x20), 1,1,...(x20), ..., 9,9,...(x20)]
    class_idx_np = np.repeat(np.arange(10, dtype=np.int64), 20).reshape(1, 200)
    class_indices = opset.constant(class_idx_np)

    # ── Redirect consumers ───────────────────────────────────────────
    # Floor output (class indices) -> constant class_indices
    for target_input in list(nodes['floor'].output(0).get_target_inputs()):
        target_input.replace_source_output(class_indices.output(0))

    # FloorMod output (spatial indices) -> spatial_indices
    for target_input in list(nodes['floormod'].output(0).get_target_inputs()):
        target_input.replace_source_output(spatial_indices.output(0))

    # ── Validate and save ────────────────────────────────────────────
    model.validate_nodes_and_infer_types()

    # Save backup of original
    import shutil
    backup_xml = output_dir / 'full_detection_original.xml'
    backup_bin = output_dir / 'full_detection_original.bin'
    orig_bin = str(xml_path).replace('.xml', '.bin')
    if not backup_xml.exists():
        shutil.copy2(str(xml_path), str(backup_xml))
        shutil.copy2(orig_bin, str(backup_bin))
        print(f"  Backup saved: {backup_xml}")

    # Save to temp file first (model may have memory-mapped the original .bin)
    tmp_xml = output_dir / 'full_detection_tmp.xml'
    ov.save_model(model, str(tmp_xml))
    # Replace original with optimized
    tmp_bin = output_dir / 'full_detection_tmp.bin'
    shutil.move(str(tmp_xml), str(xml_path))
    shutil.move(str(tmp_bin), orig_bin)

    print(f"  Optimized: {xml_path}")
    print("  TopK: [1,324000] k=200 -> [10,32400] k=20 per class (28ms -> ~3ms)")
    return True


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Export BEVFusion to OpenVINO IR (models + extensions)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint .pth file '
                             '(default: pretrained_models/bevfusion-det.pth)')
    parser.add_argument('--output-dir', type=str, default='openvino_model',
                        help='Output directory (default: openvino_model)')
    parser.add_argument('--resnet', action='store_true',
                        help='Use ResNet50 backbone instead of SwinTransformer. '
                             'RECOMMENDED for iGPU: ResNet50 produces a compact '
                             'convolution-only graph (~300 ops) that runs 3x faster '
                             'on Intel iGPU than SwinTransformer (~3300 ops with '
                             'attention/reshape ops). Uses ImageNet-pretrained '
                             'weights; neck is randomly initialized.')
    parser.add_argument('--export-legacy', action='store_true',
                        help='Also export legacy individual models (backbone_neck, '
                             'dtransform, depthnet, downsample, fuser, bev_decoder, '
                             'transfusion_head, detection_head). These are NOT used '
                             'by the optimized pipeline and slow down inference by '
                             'consuming GPU memory during compilation.')
    parser.add_argument('--no-quantize', action='store_true',
                        help='Skip INT8 quantization of camera_encoder_b6. '
                             'By default, the camera encoder is quantized to INT8 '
                             'using NNCF for ~44%% inference speedup.')
    parser.add_argument('--no-optimize', action='store_true',
                        help='Skip per-class TopK optimization of full_detection. '
                             'By default, the global TopK-200 is replaced with '
                             'per-class TopK-20 for ~25ms inference speedup.')
    parser.add_argument('--data-pkl', type=str,
                        default=None,
                        help='Path to dataset info pickle (for INT8 calibration)')
    parser.add_argument('--data-root', type=str,
                        default=None,
                        help='Path to dataset root directory (for INT8 calibration)')
    parser.add_argument('--num-cal-samples', type=int, default=30,
                        help='Number of calibration samples for INT8 quantization')
    args = parser.parse_args()

    print("=" * 70)
    print("BEVFusion Unified Export — Models + Extensions")
    print("=" * 70)

    # ── Load checkpoint ──────────────────────────────────────────────────
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = PROJECT_DIR / 'pretrained_models' / 'bevfusion-det.pth'

    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)
    print(f"  Loaded {len(state)} weight tensors")

    prefixes = detect_checkpoint_format(state)
    print(f"  Format: {prefixes['format']}")

    # Deterministic random seed for reproducible exports
    torch.manual_seed(42)
    np.random.seed(42)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output: {out_dir}\n")

    # ==================================================================
    # Part A: Neural Network Models (PyTorch -> ONNX -> OpenVINO)
    # ==================================================================
    print("=" * 70)
    print("Part A: Neural Network Models")
    if not args.export_legacy:
        print("  (Skipping legacy individual models — use --export-legacy to include)")
    print("=" * 70)

    # A1. Camera Backbone + Neck
    if args.resnet:
        print(f"\n[A1] Camera Backbone + Neck (ResNet50 + FPN)")
        cam_model = CameraBackboneNeckResNet()
        cam_model.eval()
        print("  Using ResNet50 backbone (ImageNet pretrained, neck randomly initialized)")
    else:
        print(f"\n[A1] Camera Backbone + Neck (SwinT + FPN)")
        cam_model = CameraBackboneNeck()
        load_camera_backbone_neck_weights(cam_model, state, prefixes)
        cam_model.eval()

    dummy_img = torch.randn(1, 3, 256, 704)
    with torch.no_grad():
        cam_out = cam_model(dummy_img)
    print(f"  Input: {dummy_img.shape} -> Output: {cam_out.shape}")
    if args.export_legacy:
        export_to_openvino(cam_model, dummy_img, 'camera_backbone_neck',
                           ['image'], ['features'], out_dir)

    dummy_img_b6 = torch.randn(6, 3, 256, 704)
    with torch.no_grad():
        cam_out_b6 = cam_model(dummy_img_b6)
    print(f"  Batch=6: {dummy_img_b6.shape} -> {cam_out_b6.shape}")
    if args.export_legacy:
        export_to_openvino(cam_model, dummy_img_b6, 'camera_backbone_neck_b6',
                       ['images'], ['features'], out_dir)

    # A2. VTransform components
    print(f"\n[A2] VTransform (DepthLSSTransform)")
    dtransform, depthnet, downsample, frustum, dx, bx, nx = \
        build_vtransform(state, prefixes)

    if args.export_legacy:
        dummy_depth = torch.randn(1, 1, 256, 704)
        export_to_openvino(dtransform, dummy_depth, 'vtransform_dtransform',
                           ['depth_input'], ['depth_features'], out_dir)
        dummy_depth_b6 = torch.randn(6, 1, 256, 704)
        export_to_openvino(dtransform, dummy_depth_b6, 'vtransform_dtransform_b6',
                           ['depth_input'], ['depth_features'], out_dir)

        dummy_feat = torch.randn(1, 320, 32, 88)
        export_to_openvino(depthnet, dummy_feat, 'vtransform_depthnet',
                           ['combined_features'], ['depth_and_features'], out_dir)
        dummy_feat_b6 = torch.randn(6, 320, 32, 88)
        export_to_openvino(depthnet, dummy_feat_b6, 'vtransform_depthnet_b6',
                           ['combined_features'], ['depth_and_features'], out_dir)

        dummy_bev = torch.randn(1, 80, 360, 360)
        export_to_openvino(downsample, dummy_bev, 'vtransform_downsample',
                           ['camera_bev'], ['camera_bev_downsampled'], out_dir)

    np.save(str(out_dir / 'frustum.npy'), frustum.numpy())
    np.save(str(out_dir / 'vtransform_dx.npy'), dx.numpy())
    np.save(str(out_dir / 'vtransform_bx.npy'), bx.numpy())
    np.save(str(out_dir / 'vtransform_nx.npy'), nx.numpy())
    print(f"  Saved frustum/vtransform npy files")

    # A3. Fuser
    print(f"\n[A3] Fuser (ConvFuser)")
    fuser = build_fuser(state, prefixes)
    dummy_fused = torch.randn(1, 336, 180, 180)
    if args.export_legacy:
        export_to_openvino(fuser, dummy_fused, 'fuser',
                           ['lidar_camera_bev'], ['fused_bev'], out_dir)

    # A4. BEV Decoder
    print(f"\n[A4] BEV Decoder (SECOND + SECONDFPN)")
    dec_backbone, dec_neck = build_decoder(state, prefixes)

    class DecoderWrapper(nn.Module):
        def __init__(self, backbone, neck):
            super().__init__()
            self.backbone = backbone
            self.neck = neck
        def forward(self, x):
            return self.neck(self.backbone(x))

    decoder = DecoderWrapper(dec_backbone, dec_neck)
    decoder.eval()
    dummy_dec = torch.randn(1, 256, 180, 180)
    if args.export_legacy:
        export_to_openvino(decoder, dummy_dec, 'bev_decoder',
                           ['fused_bev'], ['neck_features'], out_dir)

    # A5. TransFusion Head
    print(f"\n[A5] TransFusion Detection Head")
    head = build_transfusion_head(state, prefixes)

    class HeadWrapper(nn.Module):
        def __init__(self, head):
            super().__init__()
            self.head = head
        def forward(self, x):
            out = self.head(x)
            return (out['center'], out['height'], out['dim'], out['rot'],
                    out['vel'], out['heatmap'], out['query_heatmap_score'],
                    out['dense_heatmap'], out['top_proposals_index'],
                    out['top_proposals_class'])

    head_wrapper = HeadWrapper(head)
    head_wrapper.eval()
    dummy_head = torch.randn(1, 512, 180, 180)
    if args.export_legacy:
        export_to_openvino(head_wrapper, dummy_head, 'transfusion_head',
                           ['neck_features'],
                           ['center', 'height', 'dim', 'rot', 'vel', 'heatmap',
                            'query_heatmap_score', 'dense_heatmap',
                            'top_proposals_index', 'top_proposals_class'],
                           out_dir)

    # A6. Sparse Encoder Weights (raw NumPy files for the extension)
    print(f"\n[A6] Sparse Encoder Weights")
    sparse_dir = out_dir / 'sparse_encoder_weights'
    sparse_dir.mkdir(exist_ok=True)

    prefix = prefixes['lidar_encoder']
    count = 0
    for k, v in state.items():
        if k.startswith(prefix):
            key = k[len(prefix):]
            filename = key.replace('.', '_') + '.npy'
            np.save(str(sparse_dir / filename), v.numpy())
            count += 1
    print(f"  Saved {count} weight files to {sparse_dir}")

    # ==================================================================
    # Part A-merged: Merged Neural Network Models (fewer model calls)
    # ==================================================================
    print("\n" + "=" * 70)
    print("Part A-merged: Merged Models (for optimized inference)")
    print("=" * 70)

    # AM1. Merged Camera Encoder (backbone+neck + dtransform + depthnet)
    print(f"\n[AM1] Camera Encoder (merged: backbone+neck + dtransform + depthnet)")
    camera_encoder = CameraEncoderMerged(cam_model, dtransform, depthnet)
    camera_encoder.eval()
    dummy_imgs_b6 = torch.randn(6, 3, 256, 704)
    dummy_depths_b6 = torch.randn(6, 1, 256, 704)
    export_to_openvino(camera_encoder, (dummy_imgs_b6, dummy_depths_b6),
                       'camera_encoder_b6',
                       ['images', 'depth_maps'],
                       ['depth_logits', 'context_features'], out_dir,
                       use_direct=True)

    # AM2. Merged Detection pipeline (downsample + fuser + decoder, without head)
    print(f"\n[AM2] Detection pipeline (merged: downsample + fuser + decoder)")
    detection_head = DetectionHeadMerged(
        downsample, fuser, dec_backbone, dec_neck)
    detection_head.eval()
    dummy_cam_bev = torch.randn(1, 80, 360, 360)
    dummy_lidar_bev = torch.randn(1, 256, 180, 180)
    if args.export_legacy:
        export_to_openvino(detection_head, (dummy_cam_bev, dummy_lidar_bev),
                           'detection_head',
                           ['camera_bev', 'lidar_bev'],
                           ['decoder_features'],
                           out_dir,
                           use_direct=True)

    # AM3. Geometry model (frustum → 3D coords, runs on GPU)
    print(f"\n[AM3] Geometry Model (camera projection on GPU)")
    D_BINS = 118
    IMAGE_H, IMAGE_W = 256, 704
    DBOUND = [1.0, 60.0, 0.5]
    fH, fW = IMAGE_H // 8, IMAGE_W // 8   # 32, 88
    d = int((DBOUND[1] - DBOUND[0]) / DBOUND[2])   # 118

    ds = np.arange(d, dtype=np.float32) * DBOUND[2] + DBOUND[0]
    xs = np.linspace(0, IMAGE_W - 1, fW, dtype=np.float32)
    ys = np.linspace(0, IMAGE_H - 1, fH, dtype=np.float32)
    ds_g, ys_g, xs_g = np.meshgrid(ds, ys, xs, indexing='ij')
    frustum_np = np.stack([xs_g, ys_g, ds_g], axis=-1)  # [118, 32, 88, 3]

    class GeometryModel(nn.Module):
        """Geometry computation model: camera frustum → 3D lidar coords.

        Bakes the constant frustum as a buffer. Takes per-sample camera
        calibration matrices as input. Runs on GPU via OpenVINO.

        Input:  combined_matrices [6, 14]  (packed camera parameters)
                Layout per camera: [aug_inv_00, aug_inv_01, aug_inv_10, aug_inv_11,
                                   aug_tx, aug_ty, fx, fy, cx, cy,
                                   c2l_t_x, c2l_t_y, c2l_t_z, unused]
                camera2lidar_rot [6, 3, 3]
        Output: geom [6, N, 3] where N = D*fH*fW = 332288
        """
        def __init__(self, frustum_tensor):
            super().__init__()
            pts = frustum_tensor.reshape(-1, 3)  # [N, 3]
            self.register_buffer('pts_xy', pts[:, :2].clone())   # [N, 2]
            self.register_buffer('pts_d', pts[:, 2].clone())     # [N]

        def forward(self, combined_matrices, camera2lidar_rot):
            # combined_matrices: [6, 14]
            # camera2lidar_rot: [6, 3, 3]
            num_cams = combined_matrices.shape[0]
            N = self.pts_xy.shape[0]

            # Unpack camera parameters
            aug_inv = combined_matrices[:, :4].reshape(num_cams, 2, 2)  # [6,2,2]
            aug_t = combined_matrices[:, 4:6]                            # [6,2]
            fx = combined_matrices[:, 6:7]                               # [6,1]
            fy = combined_matrices[:, 7:8]                               # [6,1]
            cx = combined_matrices[:, 8:9]                               # [6,1]
            cy = combined_matrices[:, 9:10]                              # [6,1]
            t_vec = combined_matrices[:, 10:13]                          # [6,3]

            R = camera2lidar_rot                                         # [6,3,3]

            # Undo augmentation: xy' = aug_inv @ (pixel_xy - aug_t)
            xy = self.pts_xy.unsqueeze(0) - aug_t.unsqueeze(1)     # [6, N, 2]
            xy = torch.matmul(xy, aug_inv.transpose(1, 2))         # [6, N, 2]

            # Unproject to camera coords
            depth = self.pts_d.unsqueeze(0).expand(num_cams, -1)   # [6, N]
            cam_x = (xy[:, :, 0] - cx) * depth / fx               # [6, N]
            cam_y = (xy[:, :, 1] - cy) * depth / fy               # [6, N]
            cam_pts = torch.stack([cam_x, cam_y, depth], dim=-1)   # [6, N, 3]

            # Camera → lidar transform
            pts_3d = torch.matmul(cam_pts, R.transpose(1, 2)) + t_vec.unsqueeze(1)
            return pts_3d.reshape(-1, 3)  # [6*N, 3] = [1993728, 3]

    frustum_tensor = torch.from_numpy(frustum_np).float()
    geom_model = GeometryModel(frustum_tensor)
    geom_model.eval()

    N_pts = D_BINS * fH * fW  # 332288
    dummy_combined = torch.randn(6, 14)
    dummy_rot = torch.randn(6, 3, 3)
    export_to_openvino(geom_model, (dummy_combined, dummy_rot),
                       'geometry',
                       ['combined_matrices', 'camera2lidar_rot'],
                       ['geometry_points'],
                       out_dir,
                       use_direct=True)

    # AM4. Full Detection (detection_head + transfusion_head merged)
    print(f"\n[AM4] Full Detection (merged: downsample+fuser+decoder+transfusion)")

    class FullDetectionModel(nn.Module):
        """Fully merged detection: downsample + fuser + decoder + TransFusion head.

        Eliminates GPU→CPU→GPU round-trip between detection_head and transfusion_head.
        Input:  camera_bev [1,80,360,360], lidar_bev [1,256,180,180]
        Output: 10 detection tensors
        """
        def __init__(self, downsample, fuser, decoder_backbone, decoder_neck, transfusion_head):
            super().__init__()
            self.downsample = downsample
            self.fuser = fuser
            self.decoder_backbone = decoder_backbone
            self.decoder_neck = decoder_neck
            self.transfusion = transfusion_head

        def forward(self, camera_bev, lidar_bev):
            cam_ds = self.downsample(camera_bev)
            fused = torch.cat([cam_ds, lidar_bev], dim=1)
            fused = self.fuser(fused)
            dec_outs = self.decoder_backbone(fused)
            decoded = self.decoder_neck(dec_outs)
            out = self.transfusion(decoded)
            return (out['center'], out['height'], out['dim'], out['rot'],
                    out['vel'], out['heatmap'], out['query_heatmap_score'],
                    out['dense_heatmap'], out['top_proposals_index'],
                    out['top_proposals_class'])

    full_det = FullDetectionModel(downsample, fuser, dec_backbone, dec_neck, head)
    full_det.eval()
    try:
        export_to_openvino(full_det, (dummy_cam_bev, dummy_lidar_bev),
                           'full_detection',
                           ['camera_bev', 'lidar_bev'],
                           ['center', 'height', 'dim', 'rot', 'vel', 'heatmap',
                            'query_heatmap_score', 'dense_heatmap',
                            'top_proposals_index', 'top_proposals_class'],
                           out_dir,
                           use_direct=True)
    except Exception as e:
        print(f"  Full detection merge failed ({e}), keeping separate models")

    # ==================================================================
    # Part B: OpenVINO Extension Models (custom ops)
    # ==================================================================
    print("\n" + "=" * 70)
    print("Part B: OpenVINO Extension Models")
    print("=" * 70)
    export_extension_models(out_dir, sparse_dir)

    # ==================================================================
    # Part D: Post-Export Optimizations
    # ==================================================================
    print("\n" + "=" * 70)
    print("Part D: Post-Export Optimizations")
    print("=" * 70)

    # D1. Per-class TopK optimization for full_detection
    if not args.no_optimize:
        print(f"\n[D1] Per-class TopK optimization (full_detection)")
        if optimize_full_detection_topk(out_dir):
            print("  Done — full_detection TopK optimized")
        else:
            print("  Skipped — see messages above")
    else:
        print(f"\n[D1] Skipping TopK optimization (--no-optimize)")

    # D2. INT8 quantization for camera_encoder_b6
    if not args.no_quantize:
        print(f"\n[D2] INT8 Quantization (camera_encoder_b6)")
        if quantize_camera_encoder_int8(out_dir, args.data_pkl,
                                         args.data_root,
                                         args.num_cal_samples):
            print("  Done — camera_encoder_b6 quantized to INT8")
        else:
            print("  Skipped — see messages above")
    else:
        print(f"\n[D2] Skipping INT8 quantization (--no-quantize)")

    # ==================================================================
    # Part E: Config
    # ==================================================================
    print("\n" + "=" * 70)
    print("Part E: Configuration")
    print("=" * 70)

    config = {
        'pipeline': 'full_fusion',
        'image_size': [256, 704],
        'feature_size': [32, 88],
        'num_cameras': 6,
        'camera_names': [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
        ],
        'image_normalize': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
        },
        'resize_lim': [0.48, 0.48],
        'voxel_size': [0.075, 0.075, 0.2],
        'point_cloud_range': [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
        'sparse_shape': [1440, 1440, 41],
        'bev_size': [180, 180],
        'depth': {'D': 118, 'dbound': [1.0, 60.0, 0.5], 'C': 80},
        'vtransform': {
            'xbound': [-54.0, 54.0, 0.3],
            'ybound': [-54.0, 54.0, 0.3],
            'zbound': [-10.0, 10.0, 20.0],
            'nx': [360, 360, 1],
        },
        'num_classes': 10,
        'class_names': [
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
        ],
        'models': {
            'camera_backbone_neck': 'camera_backbone_neck.xml',
            'camera_backbone_neck_b6': 'camera_backbone_neck_b6.xml',
            'vtransform_dtransform': 'vtransform_dtransform.xml',
            'vtransform_dtransform_b6': 'vtransform_dtransform_b6.xml',
            'vtransform_depthnet': 'vtransform_depthnet.xml',
            'vtransform_depthnet_b6': 'vtransform_depthnet_b6.xml',
            'vtransform_downsample': 'vtransform_downsample.xml',
            'fuser': 'fuser.xml',
            'bev_decoder': 'bev_decoder.xml',
            'transfusion_head': 'transfusion_head.xml',
            'bev_pool': 'bev_pool.xml',
            'voxelize': 'voxelize.xml',
            'sparse_encoder': 'sparse_encoder.xml',
            'camera_encoder_b6': 'camera_encoder_b6.xml',
            'detection_head': 'detection_head.xml',
        },
        'post_processing': {
            'score_threshold': 0.0,
            'post_center_range': [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            'out_size_factor': 8,
        },
    }

    config_path = out_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: {config_path}")

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 70)
    print("Export Complete!")
    print("=" * 70)
    print(f"\nOutput directory: {out_dir}")
    print("\nExported files:")
    for f in sorted(out_dir.iterdir()):
        if f.is_dir():
            n_files = sum(1 for _ in f.iterdir())
            print(f"  {f.name + '/':50s} ({n_files} files)")
        elif f.is_file():
            size = f.stat().st_size
            if size > 1024 * 1024:
                print(f"  {f.name:50s} {size / 1024 / 1024:.1f} MB")
            elif size > 1024:
                print(f"  {f.name:50s} {size / 1024:.1f} KB")
            else:
                print(f"  {f.name:50s} {size} B")


if __name__ == '__main__':
    main()
