# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Reimplementation of FlashOCC under Apache-2.0-compatible terms.

import numpy as np
import torch
from torch import nn

from .layers import BaseModule, ConvModule


nusc_class_frequencies = np.array([
    944004, 1897170, 152386, 2391677, 16957802, 724139, 189027, 2074468,
    413451, 2384460, 5916653, 175883646, 4275424, 51393615, 61411620,
    105975596, 116424404, 1892500630,
])


class BEVOCCHead2D(BaseModule):
    def __init__(
        self,
        in_dim=256,
        out_dim=256,
        Dz=16,
        use_mask=True,
        num_classes=18,
        use_predicter=True,
        class_balance=False,
        loss_occ=None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz
        out_channels = out_dim if use_predicter else num_classes * Dz
        self.final_conv = ConvModule(
            self.in_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type="Conv2d"),
        )
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, num_classes * Dz),
            )

        self.use_mask = use_mask
        self.num_classes = num_classes
        self.class_balance = class_balance
        if self.class_balance:
            class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_classes] + 0.001))
            self.register_buffer("cls_weights", class_weights)
        self.loss_occ = None

    def forward(self, img_feats):
        occ_pred = self.final_conv(img_feats).permute(0, 3, 2, 1)
        bs, dx, dy = occ_pred.shape[:3]
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
            occ_pred = occ_pred.view(bs, dx, dy, self.Dz, self.num_classes)
        return occ_pred

    def get_occ(self, occ_pred, img_metas=None):
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)
        return list(occ_res)

    def get_occ_gpu(self, occ_pred, img_metas=None):
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1).int()
        return list(occ_res)
