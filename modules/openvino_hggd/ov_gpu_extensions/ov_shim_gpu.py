# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
OpenVINO GPU Extension shim for HGGD using native GPU kernels with FP16.

Replaces pytorch3d ops with native GPU OpenVINO custom layer implementations.
Uses the v2 kernels which support FP16 data correctly via INPUT0_TYPE macros.

Usage:
    from ov_gpu_extensions.ov_shim_gpu import install_ov_shim
    install_ov_shim()
"""
import sys
import types
from pathlib import Path
from typing import Optional, NamedTuple

import torch
import numpy as np

# Import native GPU ops
try:
    from .pointcloud_ops_native_gpu import create_ops, NativePointCloudOpsGPU
except ImportError:
    from pointcloud_ops_native_gpu import create_ops, NativePointCloudOpsGPU


_ops = None


def get_ops() -> NativePointCloudOpsGPU:
    global _ops
    if _ops is None:
        _ops = create_ops(precision='f16')  # Use FP16 for performance
    return _ops


def install_ov_shim(verbose: bool = True, precision: str = 'f16'):
    """Install GPU OpenVINO extension shim as pytorch3d replacement.
    
    Args:
        verbose: Print setup info
        precision: 'f16' (default, fast), 'f32' (accurate), or 'default' (GPU decides)
    """
    global _ops
    _ops = create_ops(precision=precision)
    
    if verbose:
        ext_path = Path(__file__).parent / 'build' / 'libopenvino_pointcloud_extension.so'
        print(f"[OV GPU Shim] Extension: {ext_path.name}")
        print(f"[OV GPU Shim] Precision: {precision}")
    
    # Create fake pytorch3d modules
    pytorch3d_ops = types.ModuleType('pytorch3d.ops')
    pytorch3d_ops.knn_points = knn_points
    pytorch3d_ops.ball_query = ball_query
    pytorch3d_ops.sample_farthest_points = sample_farthest_points
    
    pytorch3d_ops_utils = types.ModuleType('pytorch3d.ops.utils')
    pytorch3d_ops_utils.masked_gather = masked_gather
    
    pytorch3d_transforms = types.ModuleType('pytorch3d.transforms')
    pytorch3d_transforms.euler_angles_to_matrix = euler_angles_to_matrix
    pytorch3d_transforms.matrix_to_quaternion = matrix_to_quaternion
    
    pytorch3d = types.ModuleType('pytorch3d')
    pytorch3d.ops = pytorch3d_ops
    pytorch3d.transforms = pytorch3d_transforms
    
    sys.modules['pytorch3d'] = pytorch3d
    sys.modules['pytorch3d.ops'] = pytorch3d_ops
    sys.modules['pytorch3d.ops.utils'] = pytorch3d_ops_utils
    sys.modules['pytorch3d.transforms'] = pytorch3d_transforms
    
    if verbose:
        print("[OV GPU Shim] pytorch3d shim installed (GPU)")


# ═══════════════════════════════════════════════════════════════════════════════
# Point Cloud Operations (pytorch3d API compatible)
# ═══════════════════════════════════════════════════════════════════════════════

class KNNResult(NamedTuple):
    dists: torch.Tensor
    idx: torch.Tensor
    knn: Optional[torch.Tensor] = None


def knn_points(
    p1: torch.Tensor,
    p2: torch.Tensor,
    K: int = 1,
    norm: int = 2,
    return_nn: bool = False,
    return_sorted: bool = True,
    **kwargs
) -> KNNResult:
    """KNN using native GPU extension."""
    ops = get_ops()
    
    p1_np = p1.detach().cpu().numpy().astype(np.float32)
    p2_np = p2.detach().cpu().numpy().astype(np.float32)
    
    dists, idx = ops.knn_points(p1_np, p2_np, k=K)
    
    dists_t = torch.from_numpy(dists).to(p1.device, dtype=p1.dtype)
    idx_t = torch.from_numpy(idx).to(p1.device, dtype=torch.int64)
    
    knn = None
    if return_nn:
        knn = masked_gather(p2, idx_t)
    
    return KNNResult(dists=dists_t, idx=idx_t, knn=knn)


def ball_query(
    p1: torch.Tensor,
    p2: torch.Tensor,
    K: int,
    radius: float,
    return_nn: bool = False,
    **kwargs
):
    """Ball query using native GPU extension."""
    ops = get_ops()
    
    p1_np = p1.detach().cpu().numpy().astype(np.float32)
    p2_np = p2.detach().cpu().numpy().astype(np.float32)
    
    dists, idx = ops.ball_query(p1_np, p2_np, k=K, radius=radius)
    
    dists_t = torch.from_numpy(dists).to(p1.device, dtype=p1.dtype)
    idx_t = torch.from_numpy(idx).to(p1.device, dtype=torch.int64)
    
    if return_nn:
        knn = masked_gather(p2, idx_t)
        return dists_t, idx_t, knn
    
    return dists_t, idx_t


def sample_farthest_points(
    points: torch.Tensor,
    K: int = None,
    lengths: torch.Tensor = None,
    random_start_point: bool = False,
    **kwargs
):
    """Farthest point sampling using native GPU extension.
    
    Uses FPSWithLengths GPU kernel for variable-length batches - no Python loops.
    
    Args:
        points: [B, N, 3] point clouds (may be zero-padded)
        K: Number of points to sample per batch
        lengths: [B] actual valid lengths for each batch (if provided)
        random_start_point: Whether to start from random point (currently ignored)
    
    Returns:
        sampled: [B, K, 3] sampled points
        idx: [B, K] indices of sampled points
    """
    ops = get_ops()
    
    pts_np = points.detach().cpu().numpy().astype(np.float32)
    
    # Pass lengths directly to GPU kernel - no Python loops
    lengths_np = None
    if lengths is not None:
        lengths_np = lengths.detach().cpu().numpy().astype(np.float32)
    
    sampled_np, idx_np = ops.fps(pts_np, k=K, lengths=lengths_np)
    
    sampled_t = torch.from_numpy(sampled_np).to(points.device, dtype=points.dtype)
    idx_t = torch.from_numpy(idx_np).to(points.device, dtype=torch.int64)
    
    return sampled_t, idx_t


def masked_gather(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by indices."""
    ops = get_ops()
    
    pts_np = points.detach().cpu().numpy().astype(np.float32)
    idx_np = idx.detach().cpu().numpy().astype(np.int32)
    
    gathered = ops.gather(pts_np, idx_np)
    
    return torch.from_numpy(gathered).to(points.device, dtype=points.dtype)


# ═══════════════════════════════════════════════════════════════════════════════
# Transform Operations (pytorch3d API compatible)
# ═══════════════════════════════════════════════════════════════════════════════

def euler_angles_to_matrix(euler: torch.Tensor, convention: str = "XYZ") -> torch.Tensor:
    """Convert Euler angles to rotation matrices (numpy fallback)."""
    if euler.dim() == 1:
        euler = euler.unsqueeze(0)
    
    device = euler.device
    dtype = euler.dtype
    e = euler.cpu().numpy()
    batch = e.shape[0]
    
    mats = np.zeros((batch, 3, 3), dtype=np.float32)
    
    for b in range(batch):
        R = np.eye(3, dtype=np.float32)
        for i, axis in enumerate(convention):
            a = e[b, i]
            c, s = np.cos(a), np.sin(a)
            if axis == 'X':
                Ri = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
            elif axis == 'Y':
                Ri = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
            else:  # Z
                Ri = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            R = R @ Ri
        mats[b] = R
    
    return torch.from_numpy(mats).to(device=device, dtype=dtype)


def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to quaternion (numpy fallback)."""
    device = R.device
    dtype = R.dtype
    
    if R.dim() == 2:
        R = R.unsqueeze(0)
    
    R_np = R.cpu().numpy()
    batch = R_np.shape[0]
    quats = np.zeros((batch, 4), dtype=np.float32)
    
    for b in range(batch):
        m = R_np[b]
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        
        quats[b] = [w, x, y, z]
    
    return torch.from_numpy(quats).to(device=device, dtype=dtype)


if __name__ == "__main__":
    print("Testing GPU shim...")
    install_ov_shim(verbose=True)
    
    import pytorch3d.ops
    
    p1 = torch.randn(1, 100, 3)
    p2 = torch.randn(1, 500, 3)
    
    result = pytorch3d.ops.knn_points(p1, p2, K=8)
    print(f"KNN result: dists {result.dists.shape}, idx {result.idx.shape}")
    
    sampled, idx = pytorch3d.ops.sample_farthest_points(p2, K=32)
    print(f"FPS result: sampled {sampled.shape}, idx {idx.shape}")
    
    print("GPU shim test passed!")
