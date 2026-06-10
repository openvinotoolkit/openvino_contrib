"""
BEV Pooling ops for FlashOCC.

bev_pool    - original BEVDet coordinate-based pool (CUDA ext or CPU fallback)
bev_pool_v2 - interval-based accumulation (bev_pool_v2 CUDA kernel or CPU fallback)
              bev_pool_v2 also carries an ONNX symbolic so that torch.onnx.export
              emits a "flashocc::FlashOCCBEVPoolV2" custom-op node in the graph,
              which the OpenVINO C++ extension can handle.
"""

import warnings

import torch
import torch.nn as nn
from torch.autograd import Function

try:
    from . import bev_pool_ext  # CUDA extension (if available)
except ImportError:
    bev_pool_ext = None
    warnings.warn(
        "BEV pool CUDA extension not available. Using CPU fallback. "
        "Performance will be significantly slower."
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _decode_coords_xyzb(coords):
    """Original BEVDet/FlashOCC convention: coords = [x, y, z, b]"""
    x = coords[:, 0].long()
    y = coords[:, 1].long()
    z = coords[:, 2].long()
    b = coords[:, 3].long()
    return x, y, z, b


def _build_intervals_from_sorted_ranks(ranks_bev_sorted):
    """Given a sorted ranks_bev tensor, build interval starts and lengths."""
    if ranks_bev_sorted.numel() == 0:
        dev = ranks_bev_sorted.device
        return (
            torch.zeros((0,), dtype=torch.long, device=dev),
            torch.zeros((0,), dtype=torch.long, device=dev),
        )
    change = torch.ones_like(ranks_bev_sorted, dtype=torch.bool)
    change[1:] = ranks_bev_sorted[1:] != ranks_bev_sorted[:-1]
    interval_starts = torch.where(change)[0].long()
    interval_ends = torch.cat([
        interval_starts[1:],
        torch.tensor([ranks_bev_sorted.numel()], device=ranks_bev_sorted.device),
    ])
    interval_lengths = (interval_ends - interval_starts).long()
    return interval_starts, interval_lengths


# ── bev_pool (coordinate-based) ────────────────────────────────────────────────

def bev_pool_cpu_sum(feats, coords, B, D, H, W):
    C = feats.shape[1]
    out = feats.new_zeros((B, D, H, W, C))
    flat_out = out.view(-1, C)
    x, y, z, b = _decode_coords_xyzb(coords)
    valid = (b >= 0) & (b < B) & (z >= 0) & (z < D) & (y >= 0) & (y < H) & (x >= 0) & (x < W)
    if valid.any():
        lin = (((b[valid] * D + z[valid]) * H + y[valid]) * W + x[valid]).long()
        flat_out.index_add_(0, lin, feats[valid])
    return out.permute(0, 4, 1, 2, 3).contiguous()  # [B, C, D, H, W]


class QuickBevPoolingCPU(Function):
    @staticmethod
    def forward(ctx, feats, coords, ranks, B, D, H, W):
        ctx.save_for_backward(coords)
        ctx.B, ctx.D, ctx.H, ctx.W = int(B), int(D), int(H), int(W)
        return bev_pool_cpu_sum(feats, coords, B, D, H, W)

    @staticmethod
    def backward(ctx, grad_output):
        (coords,) = ctx.saved_tensors
        x, y, z, b = _decode_coords_xyzb(coords)
        valid = (
            (b >= 0) & (b < ctx.B) & (z >= 0) & (z < ctx.D) &
            (y >= 0) & (y < ctx.H) & (x >= 0) & (x < ctx.W)
        )
        N, C = coords.shape[0], grad_output.shape[1]
        grad_feats = grad_output.new_zeros((N, C))
        if valid.any():
            grad_feats[valid] = grad_output[b[valid], :, z[valid], y[valid], x[valid]]
        return grad_feats, None, None, None, None, None, None


def bev_pool(feats, coords, B, D, H, W, pooling_method='sum'):
    """BEV pooling (coordinate-based) with CUDA or CPU fallback."""
    assert feats.shape[0] == coords.shape[0]
    ranks = (
        coords[:, 0] * (H * D * B) +
        coords[:, 1] * (D * B) +
        coords[:, 2] * B +
        coords[:, 3]
    )
    _, indices = torch.sort(ranks)
    feats, coords, ranks = feats[indices], coords[indices], ranks[indices]

    if bev_pool_ext is not None and feats.is_cuda:
        kept = torch.ones(feats.shape[0], device=feats.device, dtype=torch.bool)
        kept[1:] = ranks[1:] != ranks[:-1]
        interval_starts = torch.where(kept)[0].int()
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = feats.shape[0] - interval_starts[-1]
        coords = coords.int()
        if pooling_method == 'sum':
            out = bev_pool_ext.bev_sum_pool_forward(
                feats, coords, interval_lengths, interval_starts, B, D, H, W)
        elif pooling_method == 'max':
            out = bev_pool_ext.bev_max_pool_forward(
                feats, coords, interval_lengths, interval_starts, B, D, H, W)
        else:
            raise ValueError(f"Unsupported pooling method: {pooling_method}")
        return out.permute(0, 4, 1, 2, 3).contiguous()
    else:
        return QuickBevPoolingCPU.apply(feats, coords, ranks, B, D, H, W)


# ── bev_pool_v2 (interval-based, with ONNX symbolic) ──────────────────────────

def _bev_pool_v2_cpu(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                     B, Z, Y, X, C, interval_starts, interval_lengths):
    """Pure-Python CPU implementation of bev_pool_v2."""
    out = feat.new_zeros((B, Z, Y, X, C))
    flat_out = out.view(-1, C)         # [B*Z*Y*X, C]
    depth_flat = depth.reshape(-1)     # [Md]
    feat_flat  = feat.reshape(-1, C)   # [Mf, C]

    # Re-sort if ranks_bev is not sorted
    if ranks_bev.numel() > 1 and torch.any(ranks_bev[1:] < ranks_bev[:-1]):
        _, order = torch.sort(ranks_bev)
        ranks_bev   = ranks_bev[order]
        ranks_feat  = ranks_feat[order]
        ranks_depth = ranks_depth[order]
        interval_starts, interval_lengths = _build_intervals_from_sorted_ranks(ranks_bev)

    if interval_starts is None or interval_starts.numel() == 0:
        interval_starts, interval_lengths = _build_intervals_from_sorted_ranks(ranks_bev)

    K = int(interval_starts.numel())
    for i in range(K):
        s = int(interval_starts[i].item())
        l = int(interval_lengths[i].item())
        if l <= 0:
            continue
        ids = torch.arange(s, s + l, device=feat.device, dtype=torch.long)
        rf  = ranks_feat[ids].long()
        rd  = ranks_depth[ids].long()
        rb  = ranks_bev[ids].long()
        contrib = feat_flat[rf] * depth_flat[rd].unsqueeze(1)
        flat_out.index_add_(0, rb, contrib)

    return out.permute(0, 4, 1, 2, 3).contiguous()  # [B, C, Z, Y, X]


class _BEVPoolV2Fn(Function):
    """
    torch.autograd.Function for bev_pool_v2.
    forward:  CPU (or CUDA ext) interval-based accumulation.
    symbolic: emits flashocc::FlashOCCBEVPoolV2 custom op in ONNX graph.
    """

    @staticmethod
    def forward(ctx, depth, feat, ranks_depth, ranks_feat, ranks_bev,
                interval_starts, interval_lengths, B, Z, Y, X, C):
        # CUDA extension path
        if bev_pool_ext is not None and hasattr(bev_pool_ext, "bev_pool_v2_forward"):
            return bev_pool_ext.bev_pool_v2_forward(
                depth, feat, ranks_depth, ranks_feat, ranks_bev,
                (B, Z, Y, X, C), interval_starts, interval_lengths)

        # CPU fallback
        return _bev_pool_v2_cpu(
            depth, feat,
            ranks_depth, ranks_feat, ranks_bev,
            B, Z, Y, X, C,
            interval_starts, interval_lengths,
        )

    @staticmethod
    def symbolic(g, depth, feat, ranks_depth, ranks_feat, ranks_bev,
                 interval_starts, interval_lengths, B, Z, Y, X, C):
        """
        ONNX symbolic: emit a custom FlashOCCBEVPoolV2 op node.
        ranks_* and interval_* are embedded as constant initializers.
        B, Z, Y, X are op attributes.
        """
        return g.op(
            "flashocc::FlashOCCBEVPoolV2",
            depth, feat,
            ranks_depth, ranks_feat, ranks_bev,
            interval_starts, interval_lengths,
            B_i=int(B),
            Z_i=int(Z),
            Y_i=int(Y),
            X_i=int(X),
        )


def bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
    """
    Interval-based BEV accumulation matching the bev_pool_v2 CUDA kernel.

    Args:
        depth:            [B*N*D*H*W]       depth probs (may be flat or shaped)
        feat:             [B*N*H*W, C]      image features
        ranks_depth:      [M]               depth index per valid point
        ranks_feat:       [M]               feature index per valid point
        ranks_bev:        [M]               flat BEV voxel index
        bev_feat_shape:   (B, Z, Y, X, C)  output shape tuple
        interval_starts:  [K]               start of each BEV voxel interval
        interval_lengths: [K]               length of each BEV voxel interval

    Returns:
        Tensor [B, C, Z, Y, X]
    """
    B, Z, Y, X, C = [int(v) for v in bev_feat_shape]

    # Ensure integer dtypes for index tensors
    ranks_depth = ranks_depth.int()
    ranks_feat  = ranks_feat.int()
    ranks_bev   = ranks_bev.int()
    if interval_starts is not None:
        interval_starts  = interval_starts.int()
        interval_lengths = interval_lengths.int()
    else:
        interval_starts, interval_lengths = _build_intervals_from_sorted_ranks(
            ranks_bev.long())
        interval_starts  = interval_starts.int()
        interval_lengths = interval_lengths.int()

    return _BEVPoolV2Fn.apply(
        depth, feat,
        ranks_depth, ranks_feat, ranks_bev,
        interval_starts, interval_lengths,
        B, Z, Y, X, C,
    )


__all__ = ["bev_pool", "bev_pool_v2"]
