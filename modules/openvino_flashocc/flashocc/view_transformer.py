# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_3x3_onnx(m: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    a, b, c = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    d, e, f = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    g, h, i = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    A = e * i - f * h
    B_ = -(d * i - f * g)
    C = d * h - e * g
    D_ = -(b * i - c * h)
    E = a * i - c * g
    F_ = -(a * h - b * g)
    G = b * f - c * e
    H = -(a * f - c * d)
    I = a * e - b * d
    det = a * A + b * B_ + c * C
    det = torch.where(det.abs() < eps, det + eps, det)
    inv = torch.stack([A, D_, G, B_, E, H, C, F_, I], dim=-1).reshape(m.shape)
    return inv * (1.0 / det).unsqueeze(-1).unsqueeze(-1)


def _bev_pool_v2_fallback(depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape):
    b, dz, dy, dx, c = [int(v) for v in bev_feat_shape]
    depth_flat = depth.reshape(-1)
    feat_flat = feat.reshape(-1, c)
    weighted = depth_flat[ranks_depth.long()].unsqueeze(1) * feat_flat[ranks_feat.long()]
    out = feat.new_zeros((b * dz * dy * dx, c))
    idx = ranks_bev.long().unsqueeze(1).expand(-1, c)
    out = out.scatter_add(0, idx, weighted)
    return out.view(b, dz, dy, dx, c).permute(0, 4, 1, 2, 3).contiguous()


class LSSViewTransformer(nn.Module):
    def __init__(
        self,
        grid_config,
        input_size,
        downsample=16,
        in_channels=512,
        out_channels=64,
        accelerate=False,
        sid=False,
        collapse_z=True,
    ):
        super().__init__()
        self.grid_config = grid_config
        self.downsample = downsample
        self.create_grid_infos(**grid_config)
        self.sid = sid
        self.frustum = self.create_frustum(grid_config["depth"], input_size, downsample)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.depth_net = nn.Conv2d(in_channels, self.D + self.out_channels, kernel_size=1, padding=0)
        self.accelerate = accelerate
        self.initial_flag = True
        self.collapse_z = collapse_z

    def create_grid_infos(self, x, y, z, **kwargs):
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2] for cfg in [x, y, z]])

    def create_frustum(self, depth_cfg, input_size, downsample):
        h_in, w_in = input_size
        h_feat, w_feat = h_in // downsample, w_in // downsample
        d = torch.arange(*depth_cfg, dtype=torch.float).view(-1, 1, 1).expand(-1, h_feat, w_feat)
        self.D = d.shape[0]
        if self.sid:
            d_sid = torch.arange(self.D).float()
            depth_cfg_t = torch.tensor(depth_cfg).float()
            d_sid = torch.exp(torch.log(depth_cfg_t[0]) + d_sid / (self.D - 1) *
                              torch.log((depth_cfg_t[1] - 1) / depth_cfg_t[0]))
            d = d_sid.view(-1, 1, 1).expand(-1, h_feat, w_feat)
        x = torch.linspace(0, w_in - 1, w_feat, dtype=torch.float).view(1, 1, w_feat).expand(self.D, h_feat, w_feat)
        y = torch.linspace(0, h_in - 1, h_feat, dtype=torch.float).view(1, h_feat, 1).expand(self.D, h_feat, w_feat)
        return torch.stack((x, y, d), -1)

    def get_ego_coor(self, sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda):
        b, n, _, _ = sensor2ego.shape
        points = self.frustum.to(sensor2ego) - post_trans.view(b, n, 1, 1, 1, 3)
        points = _inverse_3x3_onnx(post_rots).view(b, n, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        points = torch.cat((points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        combine = sensor2ego[:, :, :3, :3].matmul(_inverse_3x3_onnx(cam2imgs))
        points = combine.view(b, n, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += sensor2ego[:, :, :3, 3].view(b, n, 1, 1, 1, 3)
        points = bda.view(b, 1, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        return points

    def init_acceleration_v2(self, coor):
        ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths = self.voxel_pooling_prepare_v2(coor)
        self.ranks_bev = ranks_bev.int().contiguous()
        self.ranks_feat = ranks_feat.int().contiguous()
        self.ranks_depth = ranks_depth.int().contiguous()
        self.interval_starts = interval_starts.int().contiguous()
        self.interval_lengths = interval_lengths.int().contiguous()

    def voxel_pooling_prepare_v2(self, coor):
        b, n, d, h, w, _ = coor.shape
        num_points = b * n * d * h * w
        ranks_depth = torch.arange(0, num_points, dtype=torch.int, device=coor.device)
        ranks_feat = torch.arange(0, num_points // d, dtype=torch.int, device=coor.device)
        ranks_feat = ranks_feat.reshape(b, n, 1, h, w).expand(b, n, d, h, w).flatten()

        coor = ((coor - self.grid_lower_bound.to(coor)) / self.grid_interval.to(coor))
        coor = coor.long().reshape(-1, 3)
        points_per_batch = n * d * h * w
        batch_idx = torch.div(ranks_depth, points_per_batch, rounding_mode="floor").reshape(-1, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)

        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])
        if kept.sum() == 0:
            return None, None, None, None, None

        coor, ranks_depth, ranks_feat = coor[kept], ranks_depth[kept], ranks_feat[kept]
        ranks_bev = coor[:, 3] * (self.grid_size[2] * self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]
        order = torch.sort(ranks_bev)[1]
        ranks_bev, ranks_depth, ranks_feat = ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = torch.ones(ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return (ranks_bev.int().contiguous(), ranks_depth.int().contiguous(),
                ranks_feat.int().contiguous(), interval_starts.int().contiguous(),
                interval_lengths.int().contiguous())

    def voxel_pooling_v2(self, coor, depth, feat):
        ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths = self.voxel_pooling_prepare_v2(coor)
        if ranks_feat is None:
            dummy = torch.zeros(size=[
                feat.shape[0], feat.shape[2], int(self.grid_size[2]),
                int(self.grid_size[1]), int(self.grid_size[0])
            ]).to(feat)
            return torch.cat(dummy.unbind(dim=2), 1)
        feat = feat.permute(0, 1, 3, 4, 2)
        bev_feat_shape = (depth.shape[0], int(self.grid_size[2]), int(self.grid_size[1]),
                          int(self.grid_size[0]), feat.shape[-1])
        bev_feat = _bev_pool_v2_fallback(depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape)
        if self.collapse_z:
            bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        return bev_feat

    def pre_compute(self, input):
        if self.initial_flag:
            coor = self.get_ego_coor(*input[1:7])
            self.init_acceleration_v2(coor)
            self.initial_flag = False

    def view_transform_core(self, input, depth, tran_feat):
        b, n, c, h, w = input[0].shape
        if self.accelerate:
            feat = tran_feat.view(b, n, self.out_channels, h, w).permute(0, 1, 3, 4, 2)
            depth = depth.view(b, n, self.D, h, w)
            bev_feat_shape = (depth.shape[0], int(self.grid_size[2]), int(self.grid_size[1]),
                              int(self.grid_size[0]), feat.shape[-1])
            bev_feat = _bev_pool_v2_fallback(
                depth, feat, self.ranks_depth, self.ranks_feat, self.ranks_bev, bev_feat_shape)
            bev_feat = bev_feat.squeeze(2)
        else:
            coor = self.get_ego_coor(*input[1:7])
            bev_feat = self.voxel_pooling_v2(
                coor, depth.view(b, n, self.D, h, w),
                tran_feat.view(b, n, self.out_channels, h, w))
        return bev_feat, depth

    def view_transform(self, input, depth, tran_feat):
        if self.accelerate:
            self.pre_compute(input)
        return self.view_transform_core(input, depth, tran_feat)

    def forward(self, input):
        x = input[0]
        b, n, c, h, w = x.shape
        x = x.view(b * n, c, h, w)
        x = self.depth_net(x)
        depth_digit = x[:, :self.D, ...]
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]
        depth = depth_digit.softmax(dim=1)
        return self.view_transform(input, depth, tran_feat)

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda):
        return None
