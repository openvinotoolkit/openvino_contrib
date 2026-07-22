# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Adapted from BEVDet detector architecture (https://github.com/HuangJunJie2017/BEVDet), Apache-2.0, Copyright (c) OpenMMLab.
# Adapted from FlashOCC detector pipeline (https://github.com/Yzichen/FlashOCC), Apache-2.0, Copyright (c) Institute of Intelligent Control, Dalian University of Technology.

import torch
import torch.nn as nn

from .factories import build_backbone, build_head, build_neck

class FlashOCCModel(nn.Module):
    def __init__(
        self,
        img_backbone,
        img_neck,
        img_view_transformer,
        img_bev_encoder_backbone,
        img_bev_encoder_neck,
        occ_head=None,
        **kwargs,
    ):
        super().__init__()
        self.img_backbone = build_backbone(img_backbone)
        self.img_neck = build_neck(img_neck) if img_neck is not None else None
        self.with_img_neck = self.img_neck is not None
        self.img_view_transformer = build_neck(img_view_transformer)
        self.img_bev_encoder_backbone = build_backbone(img_bev_encoder_backbone)
        self.img_bev_encoder_neck = build_neck(img_bev_encoder_neck)
        self.occ_head = build_head(occ_head) if occ_head is not None else None

    def image_encoder(self, img, stereo=False):
        imgs = img
        b, n, c, im_h, im_w = imgs.shape
        imgs = imgs.view(b * n, c, im_h, im_w)
        x = self.img_backbone(imgs)
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            x = x[1:]
        if self.with_img_neck:
            x = self.img_neck(x)
            if isinstance(x, (list, tuple)):
                x = x[0]
        _, output_dim, output_h, output_w = x.shape
        x = x.view(b, n, output_dim, output_h, output_w)
        return x, stereo_feat

    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if isinstance(x, (list, tuple)):
            x = x[0]
        return x

    def prepare_inputs(self, inputs):
        assert len(inputs) == 7
        b, n, c, h, w = inputs[0].shape
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = inputs
        sensor2egos = sensor2egos.view(b, n, 4, 4)
        ego2globals = ego2globals.view(b, n, 4, 4)
        keyego2global = ego2globals[:, 0, ...].unsqueeze(1)
        global2keyego = torch.linalg.inv(keyego2global)
        sensor2keyegos = global2keyego @ ego2globals @ sensor2egos
        sensor2keyegos = sensor2keyegos.float()
        return [imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda]

    def extract_img_feat(self, img_inputs, img_metas=None, **kwargs):
        img_inputs = self.prepare_inputs(img_inputs)
        x, _ = self.image_encoder(img_inputs[0])
        x, depth = self.img_view_transformer([x] + img_inputs[1:7])
        x = self.bev_encoder(x)
        return [x], depth

    def extract_feat(self, points=None, img_inputs=None, img_metas=None, **kwargs):
        img_feats, depth = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        return img_feats, None, depth

