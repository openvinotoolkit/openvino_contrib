#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
SAM 3D Objects — OpenVINO Standalone Inference Script

Runs the full SAM3D pipeline using exported OpenVINO IR models and
custom C++/OpenCL extensions.

Usage:
    # kid_box  (matches NVIDIA baseline default: mask-index=0)
    python run_inference_standalone.py \\
        --model-dir ./exported_models \\
        --image notebook/images/kid_box/image.png \\
        --mask  notebook/images/kid_box/0.png \\
        --output kid_box.ply

    # RGBA PNG (mask already embedded in alpha channel)
    python run_inference_standalone.py \\
        --model-dir ./exported_models \\
        --image /tmp/input_rgba.png \\
        --output output.ply

NOTE: Always supply --mask (or an RGBA image with the object mask in the
alpha channel).  Without a mask the entire image is treated as the object,
the 'cropped' DINO slot receives the padded full image, and the voxel count
falls from ~24 k to ~9 k compared to the NVIDIA baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# ─── OpenVINO ───────────────────────────────────────────────────────────────
try:
    import openvino as ov
except ImportError:
    raise ImportError("OpenVINO Python API not found. pip install openvino>=2025.0")

# ─── SAM3D project for preprocessing utilities and MoGe ────────────────────
SAM3D_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SAM3D_ROOT))

# ─── numpy-only helpers ─────────────────────────────────────────────────────

def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _layer_norm(x: np.ndarray, weight: Optional[np.ndarray] = None,
                bias: Optional[np.ndarray] = None, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    if weight is not None:
        x_norm = x_norm * weight
    if bias is not None:
        x_norm = x_norm + bias
    return x_norm


def _linear(x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray] = None) -> np.ndarray:
    """Dense linear layer: out = x @ W^T + b."""
    out = x @ weight.T
    if bias is not None:
        out = out + bias
    return out


# ─── Image preprocessing ─────────────────────────────────────────────────────

def load_image_rgba(path: str, mask_path: Optional[str] = None) -> np.ndarray:
    """Load image as RGBA numpy array (H, W, 4) float32 in [0, 1].

    If mask_path is provided, loads image as RGB and mask separately.
    """
    from PIL import Image
    img = Image.open(path)
    if mask_path is not None:
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0    # (H, W, 3)
        mask_img = Image.open(mask_path)
        mask_arr = np.array(mask_img, dtype=np.float32)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., -1]  # take last channel
        if mask_arr.max() > 1.0:
            mask_arr = mask_arr / 255.0
        # Binary threshold
        mask_arr = (mask_arr > 0.5).astype(np.float32)
        # Resize mask if needed
        if mask_arr.shape != arr.shape[:2]:
            mask_pil = Image.fromarray((mask_arr * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((arr.shape[1], arr.shape[0]), Image.NEAREST)
            mask_arr = np.array(mask_pil, dtype=np.float32) / 255.0
        return np.concatenate([arr, mask_arr[..., None]], axis=-1)
    else:
        img = img.convert("RGBA")
        return np.array(img, dtype=np.float32) / 255.0


def get_crop_bbox(mask: np.ndarray) -> Tuple[int, int, int]:
    """Compute crop bbox matching baseline crop_around_mask_with_padding(box_size_factor=1.2).

    Returns (y1, x1, size) — a centered square crop of size max(bbox_h, bbox_w)*1.2.
    The crop may extend outside image bounds; use apply_crop() to pad accordingly.

    Mirrors compute_mask_bbox() exactly:
      - occupancy is `nonzero(mask)` (alpha != 0), NOT `mask > 0.5`
      - the corner coordinates are truncated independently via int(), so the
        resulting extent is 2*(size//2), which is one pixel smaller than `size`
        when `size` is odd.
    """
    rows = np.any(mask != 0, axis=1)
    cols = np.any(mask != 0, axis=0)
    if not np.any(rows) or not np.any(cols):
        H, W = mask.shape[:2]
        s = max(H, W)
        return 0, 0, s
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    bbox_h = rmax - rmin
    bbox_w = cmax - cmin
    size = int(max(bbox_h, bbox_w, 2) * 1.2)
    half = size // 2
    center_y = (rmin + rmax) / 2.0
    center_x = (cmin + cmax) / 2.0
    y1 = int(center_y - half)
    x1 = int(center_x - half)
    y2 = int(center_y + half)
    x2 = int(center_x + half)
    # height and width are equal by construction; use the y extent.
    return y1, x1, max(y2 - y1, x2 - x1)


def apply_crop(arr: np.ndarray, y1: int, x1: int, size: int,
               pad_val: float = 0.0, is_chw: bool = False) -> np.ndarray:
    """Apply square crop with zero-padding for out-of-bounds regions."""
    y2, x2 = y1 + size, x1 + size
    H = arr.shape[-2] if is_chw else arr.shape[0]
    W = arr.shape[-1] if is_chw else arr.shape[1]
    pt = max(0, -y1)
    pb = max(0, y2 - H)
    pl = max(0, -x1)
    pr = max(0, x2 - W)
    if pt > 0 or pb > 0 or pl > 0 or pr > 0:
        if is_chw:
            arr = np.pad(arr, ((0, 0), (pt, pb), (pl, pr)), constant_values=pad_val)
        elif arr.ndim == 3:
            arr = np.pad(arr, ((pt, pb), (pl, pr), (0, 0)), constant_values=pad_val)
        else:
            arr = np.pad(arr, ((pt, pb), (pl, pr)), constant_values=pad_val)
        y1 += pt
        x1 += pl
    y2, x2 = y1 + size, x1 + size
    if is_chw:
        return arr[:, y1:y2, x1:x2]
    return arr[y1:y2, x1:x2]


def crop_around_mask(rgb: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Crop image and mask matching baseline: box_size_factor=1.2, padding_factor=0.0."""
    y1, x1, size = get_crop_bbox(mask)
    return apply_crop(rgb, y1, x1, size), apply_crop(mask, y1, x1, size)


def pad_to_square_centered(rgb: np.ndarray, mask: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Pad both to the same square size by centering content."""
    h, w = rgb.shape[:2]
    s = max(h, w)
    pad_t = (s - h) // 2
    pad_l = (s - w) // 2
    pad_b = s - h - pad_t
    pad_r = s - w - pad_l

    rgb_sq = np.pad(rgb, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), constant_values=0.0)
    mask_sq = np.pad(mask, ((pad_t, pad_b), (pad_l, pad_r)), constant_values=0.0)
    return rgb_sq, mask_sq


def resize_to(img: np.ndarray, size: int) -> np.ndarray:
    """Bilinear resize to (size, size). img: (H, W, C) or (H, W)."""
    from PIL import Image
    if img.ndim == 2:
        im = Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))
        return np.array(im.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    else:
        im = Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))
        return np.array(im.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0


def preprocess_for_dino(rgb_hwc: np.ndarray, size: int = 518) -> np.ndarray:
    """Resize to (size, size). Returns (1, 3, H, W) float32 in [0, 1].

    NOTE: ImageNet normalization is applied INSIDE the exported DINO model
    (DinoWrapper.forward does (x - mean) / std). Feeding raw [0, 1] RGB here;
    normalizing externally too would double-normalize and corrupt the DINO
    image tokens (the SS depth-collapse bug).
    """
    resized = resize_to(rgb_hwc, size)      # (H, W, 3) in [0, 1]
    return resized.transpose(2, 0, 1)[None]  # (1, 3, H, W)


def preprocess_mask_for_dino(mask_hw: np.ndarray, size: int = 518) -> np.ndarray:
    """Resize mask. Returns (1, 1, H, W) float32.

    Baseline mask_transform is Compose([pad_to_square_centered,
    Resize(518, interpolation=0)]) — interpolation 0 is NEAREST, so the mask must
    be sampled, not interpolated.  Using bilinear (plus the uint8 round-trip in
    resize_to) blurred every mask boundary and was a significant contributor to
    the OV-vs-CUDA condition-token gap on the DINO mask blocks.
    """
    H, W = mask_hw.shape[:2]
    ys = (np.arange(size) * H / size).astype(int)
    xs = (np.arange(size) * W / size).astype(int)
    resized = np.asarray(mask_hw, dtype=np.float32)[ys[:, None], xs[None, :]]
    if resized.ndim == 2:
        resized = resized[:, :, None]       # (H, W, 1)
    return resized.transpose(2, 0, 1)[None]  # (1, 1, H, W)


def ssi_normalize_pointmap(pm_chw: np.ndarray, mask_hw: np.ndarray,
                           scale_factor: float = 1.0
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Object-centric Scale-Shift-Invariant pointmap normalization (use_scene_scale=True).

    Mirrors sam3d_objects ObjectCentricSSI._compute_scale_and_shift + _apply_metric_to_ssi:
      shift = per-channel nanmedian of masked (mask>0.5) points          (3,)
      scale = nanmedian over pixels of per-pixel max-abs of centered pts (scalar → 3,)
      normalized = (pointmap - shift) / scale

    Args:
        pm_chw:  (3, H, W) float32 pointmap (may contain NaN for invalid)
        mask_hw: (H, W) alpha/mask, values in [0, 1]
    Returns (pm_norm (3,H,W), scale (3,), shift (3,)).
    """
    C, H, W = pm_chw.shape
    flat = pm_chw.reshape(3, -1)                      # (3, H*W)
    mask_bool = mask_hw.reshape(-1) > 0.5
    mask_points = flat[:, mask_bool]                  # (3, M)

    if mask_points.size == 0 or not np.isfinite(mask_points).any():
        scale = np.full(3, scale_factor, dtype=np.float32)
        shift = np.zeros(3, dtype=np.float32)
        return pm_chw.copy(), scale, shift

    shift = np.nanmedian(mask_points, axis=1).astype(np.float32)   # (3,)
    centered = flat - shift[:, None]                               # (3, H*W)
    max_dims = np.max(np.abs(centered), axis=0)                    # (H*W,) — NaN propagates per pixel
    scale_scalar = np.nanmedian(max_dims)
    if not np.isfinite(scale_scalar) or scale_scalar == 0:
        scale_scalar = scale_factor
    scale = np.full(3, scale_scalar * scale_factor, dtype=np.float32)

    pm_norm = (pm_chw - shift[:, None, None]) / scale[:, None, None]
    return pm_norm.astype(np.float32), scale, shift


MOGE_MASK_THRESHOLD = 0.5   # MoGeModel.mask_threshold default
# Longest side fed to MoGe, or 0 to use the image's native resolution (what the
# PyTorch baseline does).  Override with SAM3D_MOGE_MAX_SIDE.
MOGE_MAX_SIDE = 0


def _moge_recover_shift(points_bhw3: np.ndarray, mask_bhw: np.ndarray,
                        downsample: int = 64) -> np.ndarray:
    """NumPy port of moge.utils.geometry_torch.recover_focal_shift (shift only).

    The exported MoGe IR only covers MoGeModel.forward(), which emits an
    affine-invariant point map whose z is defined up to an unknown shift.
    MoGeModel.infer() recovers that shift by fitting the point map against the
    normalized view-plane UVs on a 64x64 downsample. We reuse MoGe's own
    `solve_optimal_focal_shift` so the numerics match the baseline exactly.

    Args:
        points_bhw3: (B, H, W, 3) raw MoGe point map.
        mask_bhw:    (B, H, W) boolean validity mask.
    Returns: (B,) float32 z-shift.
    """
    from moge.utils.geometry_numpy import (
        solve_optimal_focal_shift, normalized_view_plane_uv_numpy,
    )

    B, H, W, _ = points_bhw3.shape
    uv = normalized_view_plane_uv_numpy(width=W, height=H)   # (H, W, 2)

    # nearest-neighbour downsample, matching F.interpolate(mode='nearest')
    ys = (np.arange(downsample) * H / downsample).astype(int)
    xs = (np.arange(downsample) * W / downsample).astype(int)
    pts_lr = points_bhw3[:, ys[:, None], xs[None, :], :]     # (B, d, d, 3)
    uv_lr = uv[ys[:, None], xs[None, :], :]                  # (d, d, 2)
    mask_lr = mask_bhw[:, ys[:, None], xs[None, :]]          # (B, d, d)

    shifts = np.zeros(B, dtype=np.float32)
    for i in range(B):
        sel = mask_lr[i]
        pts_i = pts_lr[i][sel]
        uv_i = uv_lr[sel]
        if uv_i.shape[0] < 2:
            continue
        shift_i, _focal_i = solve_optimal_focal_shift(uv_i, pts_i)
        shifts[i] = float(shift_i)
    return shifts


def pad_to_square_chw(arr_chw: np.ndarray, pad_val: float = 0.0) -> np.ndarray:
    """Pad (C, H, W) to a centered square (max(H,W)) — matches pad_to_square_centered."""
    C, h, w = arr_chw.shape
    s = max(h, w)
    pt = (s - h) // 2
    pl = (s - w) // 2
    pb = s - h - pt
    pr = s - w - pl
    return np.pad(arr_chw, ((0, 0), (pt, pb), (pl, pr)), constant_values=pad_val)



# ─── Timestep / ODE schedules ─────────────────────────────────────────────────

def shortcut_schedule(n_steps: int, rescale_t: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (ts, ds) for ShortCut ODE.

    With n_steps=25, rescale_t=3: ts warped via t/(1+(rescale_t-1)*(1-t)).
    Step size d = 0.0 (baseline always uses no_shortcut=True → d=0).
    """
    ts = np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float32)
    if rescale_t != 1.0:
        ts = ts / (1.0 + (rescale_t - 1.0) * (1.0 - ts))
    d = 0.0  # baseline uses no_shortcut=True → d=0 always
    ds = np.full(n_steps, d, dtype=np.float32)
    return ts, ds


def flowmatching_schedule(n_steps: int, rescale_t: float = 1.0) -> np.ndarray:
    """Returns timestep array [0, 1] for FlowMatching ODE (right to left for reverse)."""
    ts = np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float32)
    ts = ts / (1.0 + (rescale_t - 1.0) * (1.0 - ts))
    return ts  # [0.0, ..., 1.0]


def downsample_sparse_structure(coord_batch: np.ndarray,
                                max_coords: int = 42000,
                                downsample_factor: int = 2
                                ) -> Tuple[np.ndarray, int]:
    """NumPy port of the baseline's inference_utils.downsample_sparse_structure.

    Rescales/dedups sparse-structure coords when there are more than ``max_coords``
    voxels (mesh decoding int32 limit: max(int32)/(64*768) ~= 43691). No-op below
    the threshold — the golden inputs (~11.7k voxels) never trigger it. Kept for
    exact parity with the baseline SS decode path.

    Args:
        coord_batch: (N, 4) int array, col 0 = batch idx, cols 1: = z,y,x coords.
    Returns:
        (downsampled (M, 4) int32 coords, downsample_factor applied [1 if untouched]).
    """
    if coord_batch.shape[0] <= max_coords:
        return coord_batch, 1

    coords = coord_batch[:, 1:].astype(np.float32)     # (N, 3)
    batch_indices = coord_batch[:, 0:1]                # (N, 1)

    coords_min = coords.min(axis=0)                    # (3,)
    coords_max = coords.max(axis=0)                    # (3,)
    original_size = coords_max - coords_min + 1

    target_size = original_size / downsample_factor
    offset = (original_size - target_size) / 2.0
    target_min = coords_min + offset
    target_max = coords_min + offset + target_size - 1

    coords_normalized = (coords - coords_min) / (coords_max - coords_min)
    coords_rescaled = coords_normalized * (target_size - 1) + target_min
    coords_rescaled = np.round(coords_rescaled).astype(np.int32)
    coords_rescaled = np.clip(coords_rescaled,
                              target_min.astype(np.int32),
                              target_max.astype(np.int32))

    combined = np.concatenate([batch_indices.astype(np.int32), coords_rescaled], axis=1)
    unique_combined = np.unique(combined, axis=0)      # sorted, deduped

    if unique_combined.shape[0] > max_coords:
        # Deterministic truncation; the baseline randperm-subsamples here, but this
        # branch requires > ~336k original voxels and is unreachable for real inputs.
        unique_combined = unique_combined[:max_coords]

    return unique_combined.astype(np.int32), downsample_factor


# ─── PointPatchEmbed outer computation in Python ─────────────────────────────

class PointPatchOuterPreprocess:
    """Python implementation of PointPatchEmbed.embed_pointmap_windows outer part.

    This handles:
    1. Resize XYZ map to input_size × input_size
    2. Linear remap (identity for 'linear' mode)
    3. Project via point_proj Linear: (H, W, 3) → (H, W, 512)
    4. Add invalid_xyz_token to invalid positions

    The inner_forward (windowed self-attention) is done by OpenVINO.

    Weights loaded from exported weight files at startup.
    """

    def __init__(self, point_proj_weight: np.ndarray, point_proj_bias: np.ndarray,
                 invalid_xyz_token: np.ndarray, input_size: int = 256):
        """
        Args:
            point_proj_weight: (512, 3) float32
            point_proj_bias:   (512,)   float32
            invalid_xyz_token: (512,)   float32
            input_size: resize target (256)
        """
        self.point_proj_weight = point_proj_weight
        self.point_proj_bias = point_proj_bias
        self.invalid_xyz_token = invalid_xyz_token
        self.input_size = input_size

    def __call__(self, xyz: np.ndarray, valid_mask: Optional[np.ndarray] = None
                 ) -> np.ndarray:
        """Process XYZ map → projected features ready for inner_forward.

        Args:
            xyz: (3, H, W) float32 — pointmap in camera space
            valid_mask: (1, H, W) or (H, W) bool/float — 1=valid, 0=invalid

        Returns:
            x: (1, H', W', 512) float32 — for OV inner_forward
               where H' = W' = input_size = 256
        """

        # Resize XYZ to input_size×input_size (nearest neighbor for XYZ)
        # Use PIL or numpy for resize
        xyz_resized = self._resize_nearest(xyz, self.input_size)  # (3, S, S)
        if valid_mask is not None:
            mask_resized = self._resize_nearest_mask(
                valid_mask.squeeze(), self.input_size)  # (S, S)
            invalid = mask_resized < 0.5
        else:
            invalid = np.zeros((self.input_size, self.input_size), dtype=bool)

        # Remap type='linear' → identity
        xyz_safe = xyz_resized.copy()
        xyz_safe[:, invalid] = 0.0  # zero out invalid before projection

        # Project: (3, S, S) → (S, S, 3) → linear → (S, S, 512)
        xyz_hwc = xyz_safe.transpose(1, 2, 0)  # (S, S, 3)
        x = _linear(xyz_hwc.reshape(-1, 3),
                    self.point_proj_weight,
                    self.point_proj_bias)        # (S*S, 512)
        x = x.reshape(self.input_size, self.input_size, -1)  # (S, S, 512)

        # Add invalid_xyz_token to invalid positions
        x[invalid] = 0.0
        x[invalid] += self.invalid_xyz_token[None]  # broadcast (512,) to invalid positions

        return x[None]  # (1, S, S, 512)

    @staticmethod
    def _resize_nearest(arr: np.ndarray, size: int) -> np.ndarray:
        """Resize (C, H, W) by nearest neighbor to (C, size, size)."""
        C, H, W = arr.shape
        ys = (np.arange(size) * H / size).astype(int)
        xs = (np.arange(size) * W / size).astype(int)
        return arr[:, ys[:, None], xs[None, :]]

    @staticmethod
    def _resize_nearest_mask(mask: np.ndarray, size: int) -> np.ndarray:
        """Resize (H, W) mask by nearest neighbor."""
        H, W = mask.shape
        ys = (np.arange(size) * H / size).astype(int)
        xs = (np.arange(size) * W / size).astype(int)
        return mask[ys[:, None], xs[None, :]]


# ─── EmbedderFuser positional embedding (Python) ─────────────────────────────

class EmbedderFuserPositionalEmbed:
    """Adds learned positional embeddings from EmbedderFuser.idx_emb.

    idx_emb: (max_pos_idx+1, embed_dim) — indexed by positional group index.
    positive_embed_map maps kwarg_name → pos_idx:
      'cropped' → 0,  'full' → 1  (or equivalent learned indices)
    """

    def __init__(self, idx_emb: np.ndarray,
                 positional_embed_map: Dict[str, int]):
        self.idx_emb = idx_emb       # (n_pos, D)
        self.pem = positional_embed_map

    def apply(self, tokens: np.ndarray, pos_group: str) -> np.ndarray:
        """Add positional embedding for pos_group to tokens (1, N, D)."""
        if pos_group not in self.pem:
            return tokens
        idx = self.pem[pos_group]
        if idx >= len(self.idx_emb):
            return tokens
        pos = self.idx_emb[idx]  # (D,)
        return tokens + pos[None, None, :]  # broadcast (1, 1, D)


# ─── SAM3D OpenVINO Inference Pipeline ───────────────────────────────────────

class SAM3DInference:
    """Full SAM3D pipeline running entirely via OpenVINO + Python orchestration.
    """

    def __init__(self, model_dir: str, ext_dir: str,
                 device: str = "GPU",
                 moge_checkpoint_dir: Optional[str] = None,
                 verbose: bool = True):
        self.model_dir = Path(model_dir)
        self.ext_dir = Path(ext_dir)
        self.device = device
        self.verbose = verbose

        # Load runtime config
        cfg_path = self.model_dir / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                self.cfg = json.load(f)
        else:
            raise FileNotFoundError(f"config.json not found in {model_dir}. "
                                    "Run export.py first.")

        self._log("Initializing SAM3D OpenVINO inference pipeline...")
        self._log(f"  Device: {device}")
        self._log(f"  Model dir: {model_dir}")

        # Initialize OpenVINO Core with extensions
        self.core = ov.Core()

        # Enable model caching — avoids recompilation on subsequent runs
        cache_dir = Path(model_dir) / "model_cache"
        cache_dir.mkdir(exist_ok=True)
        self.core.set_property({"CACHE_DIR": str(cache_dir)})

        # GPU device configuration: FP16 inference precision + LATENCY mode
        # This ensures the GPU plugin uses FP16 compute (2× faster than FP32)
        # and optimizes for single-stream low-latency execution.
        if device.startswith("GPU"):
            self.core.set_property(device, {
                "INFERENCE_PRECISION_HINT": ov.Type.f16,
                "PERFORMANCE_HINT": "LATENCY",
            })

        self._load_extensions()

        # Load EmbedderFuser positional embeddings + PointPatchEmbed outer weights
        # (must be before _load_models so SLAT IO sparse conv models are compiled
        # on a clean core BEFORE the GPU device is locked by any GPU-compiled model)
        self._load_embedder_weights()

        # Load all OV models (may compile on GPU — must be after SLAT IO conv init)
        self._load_models()

        # Context unification: share OV GPU plugin's cl_context with the extension.
        # Eliminates host round-trips at extension↔OV boundaries → ~3.4× faster resblocks.
        self._inject_cl_context_to_extension()

        # Load MoGe OV model for pointmap computation
        self._load_moge(moge_checkpoint_dir)

        self._log("Pipeline initialized.")

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _stage(self, msg: str):
        """Log a pipeline stage header, reporting the previous stage's wall time."""
        now = time.perf_counter()
        prev = getattr(self, "_stage_t0", None)
        if prev is not None:
            self._log(f"  [stage] {self._stage_name} took {now - prev:.2f}s")
        self._stage_t0 = now
        self._stage_name = msg.split(":")[0]
        self._log(msg)

    def _resolve_path(self, raw: Optional[str]) -> Optional[str]:
        """Rebase a path from config.json to model_dir if it doesn't exist.

        config.json stores absolute paths from the export machine.  When running
        in a different environment (e.g. Docker), the paths must be rebased to
        the mounted model_dir so the same config can be used unchanged.
        """
        if not raw:
            return raw
        p = Path(raw)
        if p.exists():
            return str(p)
        # Try: model_dir / filename
        rebased = self.model_dir / p.name
        if rebased.exists():
            return str(rebased)
        # Try: model_dir / last two path components (for sub-dirs like weights/)
        parts = p.parts
        if len(parts) >= 2:
            rebased2 = self.model_dir / parts[-2] / parts[-1]
            if rebased2.exists():
                return str(rebased2)
        return str(p)  # return original so callers can report the missing path

    def _load_extensions(self):
        """Load custom OpenVINO C++ extensions.

        Each .so is first loaded with ctypes using RTLD_GLOBAL | RTLD_NODELETE
        so that its C++ static variables (OpenCL context, std::once_flag, …)
        live in the process-wide namespace and survive subsequent dlopen() calls
        made by the OV CPU plugin.  Without this the plugin re-initialises the
        OpenCL context on every infer_request execution (6–11 s overhead each).

        This replaces the external LD_PRELOAD=/tmp/dlopen_patch.so workaround:
        the same effect is achieved entirely in Python, no shell wrapper needed.
        """
        import ctypes
        RTLD_GLOBAL   = 0x00100   # same constant on Linux x86-64
        RTLD_NODELETE = 0x01000
        RTLD_NOW      = 0x00002

        ext_so_files = list(self.ext_dir.rglob("libov_*.so")) + \
                       list(self.ext_dir.rglob("libopenvino_*_extension.so"))
        self._ext_so_handles = {}  # keep ctypes handles alive
        for so_path in ext_so_files:
            try:
                # Pre-load with RTLD_GLOBAL|RTLD_NODELETE so statics persist.
                h = ctypes.CDLL(str(so_path), mode=RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE)
                self._ext_so_handles[so_path.name] = h
                self.core.add_extension(str(so_path))
                self._log(f"  ✓ Loaded extension: {so_path.name}")
            except Exception as e:
                self._log(f"  ✗ Failed to load extension {so_path.name}: {e}")

    def _inject_cl_context_to_extension(self):
        """Share the OV GPU plugin's OpenCL context with the sparse conv extension.

        By default the extension creates its own cl_context (separate from the OV
        GPU plugin's context).  Data crossing the extension↔OV boundary must
        round-trip through host memory, causing ~3.4× resblock slowdown.

        This method:
          1. Forces OV to allocate its GPU context (compile a trivial model).
          2. Reads the cl_context pointer from OV's C API params string.
          3. Calls sam3d_inject_cl_context() on the extension .so so the
             extension will use OV's context on first real use.

        Must be called AFTER _load_extensions() and BEFORE the first extension op.
        """
        import ctypes, re, importlib.util
        if not self.device.startswith("GPU"):
            return  # Only relevant for GPU path

        # ── Step 1: force OV GPU context creation ──
        # Compile any already-registered GPU model to trigger context allocation.
        # Use a minimal model if available; otherwise skip injection gracefully.
        trigger_model = None
        for name in ("slat_input_layer", "slat_out_layer", "ss_decoder"):
            if name in self._model_paths:
                trigger_model = name
                break
        if trigger_model is None:
            self._log("  ! _inject_cl_context: no trigger model found, skipping injection")
            return

        try:
            self._log(f"  Triggering OV GPU context via {trigger_model}...")
            _ = self._get_model(trigger_model)  # forces compile on GPU
        except Exception as e:
            self._log(f"  ! _inject_cl_context: GPU model compile failed: {e}")
            return

        # ── Step 2: get OCL_CONTEXT pointer from OV C API ──
        ov_libs_dir = Path(importlib.util.find_spec("openvino").origin).parent / "libs"
        libovc_paths = list(ov_libs_dir.glob("libopenvino_c.so*"))
        if not libovc_paths:
            self._log("  ! _inject_cl_context: libopenvino_c.so not found, skipping injection")
            return
        libovc_path = str(sorted(libovc_paths)[-1])

        try:
            libovc = ctypes.CDLL(libovc_path)
            libovc.ov_core_create.restype = ctypes.c_int
            libovc.ov_core_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            libovc.ov_core_get_default_context.restype = ctypes.c_int
            libovc.ov_core_get_default_context.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                                            ctypes.POINTER(ctypes.c_void_p)]
            libovc.ov_remote_context_get_params.restype = ctypes.c_int
            libovc.ov_remote_context_get_params.argtypes = [ctypes.c_void_p,
                                                             ctypes.POINTER(ctypes.c_size_t),
                                                             ctypes.POINTER(ctypes.c_char_p)]
            libovc.ov_remote_context_free.restype = None
            libovc.ov_remote_context_free.argtypes = [ctypes.c_void_p]
            libovc.ov_core_free.restype = None
            libovc.ov_core_free.argtypes = [ctypes.c_void_p]
            libovc.ov_free.restype = None
            libovc.ov_free.argtypes = [ctypes.c_void_p]

            c_core = ctypes.c_void_p()
            libovc.ov_core_create(ctypes.byref(c_core))

            ctx_ptr = ctypes.c_void_p()
            dev_bytes = (self.device.split(".")[0] + "." + self.device.split(".")[1]
                         if "." in self.device else self.device).encode()
            libovc.ov_core_get_default_context(c_core, dev_bytes, ctypes.byref(ctx_ptr))

            size = ctypes.c_size_t()
            params_str = ctypes.c_char_p()
            libovc.ov_remote_context_get_params(ctx_ptr, ctypes.byref(size), ctypes.byref(params_str))
            params_raw = params_str.value.decode() if params_str.value else ""
            self._log(f"  OV GPU context params: {params_raw}")

            # Parse: "{CONTEXT_TYPE:OCL,OCL_CONTEXT:0x...,OCL_QUEUE:0x...}"
            m_ctx = re.search(r'OCL_CONTEXT:(0x[0-9a-fA-F]+|\d+)', params_raw)
            if m_ctx is None:
                self._log("  ! _inject_cl_context: OCL_CONTEXT not found in params, skipping")
                libovc.ov_remote_context_free(ctx_ptr)
                libovc.ov_core_free(c_core)
                return
            cl_ctx_handle = int(m_ctx.group(1), 16) if m_ctx.group(1).startswith("0x") else int(m_ctx.group(1))
            self._log(f"  OV cl_context = 0x{cl_ctx_handle:x}")

            # Also try to get OV's command queue (may be 0 before first inference)
            cl_queue_handle = 0
            m_q = re.search(r'OCL_QUEUE:(0x[0-9a-fA-F]+|\d+)', params_raw)
            if m_q:
                cl_queue_handle = int(m_q.group(1), 16) if m_q.group(1).startswith("0x") else int(m_q.group(1))
                self._log(f"  OV cl_queue = 0x{cl_queue_handle:x}")

            # ── Step 3: inject into extension ──
            ext_so = None
            for so_name, handle in self._ext_so_handles.items():
                if "sparse_conv_3d" in so_name:
                    ext_so = handle
                    break
            if ext_so is None:
                self._log("  ! _inject_cl_context: sparse_conv_3d extension not found, skipping")
                libovc.ov_remote_context_free(ctx_ptr)
                libovc.ov_core_free(c_core)
                return

            inject_fn = ext_so.sam3d_inject_cl_context
            inject_fn.restype = None
            inject_fn.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
            inject_fn(cl_ctx_handle, cl_queue_handle)  # inject context + OV's queue if available
            self._log(f"  ✓ Injected OV cl_context 0x{cl_ctx_handle:x} queue 0x{cl_queue_handle:x} into sparse_conv_3d extension")

            # Cache the prefetch function handle for async feature DMA overlap.
            try:
                pf = ext_so.sam3d_prefetch_features
                pf.restype = ctypes.c_int
                pf.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
                self._ext_prefetch_fn = pf
            except AttributeError:
                self._ext_prefetch_fn = None

            libovc.ov_remote_context_free(ctx_ptr)
            libovc.ov_core_free(c_core)
        except Exception as e:
            self._log(f"  ! _inject_cl_context failed: {e} (continuing without injection)")


    def _load_models(self):
        """Read all exported OV IR models for lazy compilation on first use."""
        self.models: Dict[str, ov.CompiledModel] = {}
        self._model_paths: Dict[str, str] = {}  # path cache for lazy compile
        self._compiled_models: Dict[str, ov.CompiledModel] = {}

        exported = self.cfg.get("exported", {})

        def _register(name: str, path: Optional[str]):
            if path is None:
                self._log(f"  ! No path for {name}")
                return
            p = Path(self._resolve_path(path))
            if not p.exists():
                self._log(f"  ✗ Model not found: {p}")
                return
            self._model_paths[name] = str(p)
            self._log(f"  ✓ Registered {name}")

        # SS models
        _register("ss_dino_image",   exported.get("ss_dino_image"))
        _register("ss_dino_mask",    exported.get("ss_dino_mask"))
        for i in range(3):
            _register(f"ss_embedder_proj_{i}",
                      exported.get(f"ss_embedder_proj_{i}")
                      or str(self.model_dir / f"ss_embedder_proj_{i}.xml"))
        _register("ss_backbone", exported.get("ss_backbone"))
        _register("ss_latent_in_all",  exported.get("ss_latent_in_all"))
        _register("ss_latent_out_all", exported.get("ss_latent_out_all"))
        _register("ss_decoder", exported.get("ss_decoder"))

        # SLAT DINOs and projections (separate)
        _register("slat_dino_0", exported.get("slat_dino_0"))
        _register("slat_dino_1", exported.get("slat_dino_1"))
        for i in range(2):
            _register(f"slat_embedder_proj_{i}", exported.get(f"slat_embedder_proj_{i}"))
        _register("slat_attn_blocks", exported.get("slat_attn_blocks"))

        # OV models added in second export pass
        _register("ss_pointpatch_outer",       exported.get("ss_pointpatch_outer"))
        _register("ss_pointpatch",
                  exported.get("ss_pointpatch") or str(self.model_dir / "ss_pointpatch.xml"))
        _register("slat_decoder_mesh_attn",    exported.get("slat_decoder_mesh_attn"))
        _register("moge",                      exported.get("moge"))

        # SLAT ODE small models (replace NumPy implementations)
        _register("slat_t_emb_all",   exported.get("slat_t_emb_all"))
        _register("slat_input_layer", exported.get("slat_input_layer"))
        _register("slat_out_layer",   exported.get("slat_out_layer"))

        # SLAT downsample/upsample (2x voxel pooling around the attention blocks)
        _register("slat_downsample",
                  exported.get("slat_downsample") or str(self.model_dir / "slat_downsample.xml"))
        _register("slat_upsample",
                  exported.get("slat_upsample") or str(self.model_dir / "slat_upsample.xml"))

        # GS decoder: single Swin-windowed OV model (replaces 14 per-block/io XMLs)
        _register("slat_dec_gs", exported.get("slat_dec_gs"))

        # SLAT IO block fused_resblock OV models (emb merged into slat_t_emb_all)
        for bn in ["input_blocks_0", "input_blocks_1", "out_blocks_0", "out_blocks_1"]:
            key = f"slat_io_{bn}_fused_resblock"
            _register(key, exported.get(key))

        # Mesh decoder tail: sparse subdivide upsample + FlexiCubes extraction
        # (custom OV extension ops).  Optional — mesh extraction is skipped when
        # these have not been exported.
        for _mesh_name in ("slat_mesh_upsample", "slat_flexicubes"):
            _mp = exported.get(_mesh_name) or str(self.model_dir / f"{_mesh_name}.xml")
            if Path(self._resolve_path(_mp)).exists():
                _register(_mesh_name, _mp)
            else:
                self._log(f"  – {_mesh_name} not exported (mesh extraction disabled)")

    # All models run on the target device to minimize CPU↔GPU transfers.

    def _get_model(self, name: str) -> ov.CompiledModel:
        """Lazy-compile an OV model on first use.

        All models are compiled on the target device (typically GPU) to keep
        data on-device and eliminate CPU↔GPU data transfer overhead.
        Models with custom ops (SparseConv3dEngine) compile on CPU since
        the extension's evaluate() handles GPU dispatch internally via OpenCL.
        """
        if name not in self._compiled_models:
            path = self._model_paths.get(name)
            if path is None:
                raise RuntimeError(f"OV model '{name}' is not registered.")
            # Custom-op models must run on CPU (evaluate() does GPU internally)
            device = "CPU" if ("fused_resblock" in name
                               or name in ("slat_mesh_upsample", "slat_flexicubes")) else self.device
            self._log(f"  [compile] {name} → {device}...")
            compiled = self.core.compile_model(path, device)
            self._compiled_models[name] = compiled
        return self._compiled_models[name]

    def _get_moge_model(self, shape) -> ov.CompiledModel:
        """Compile MoGe for one concrete input shape.

        moge.xml is exported with a fully dynamic input.  The Arc GPU plugin
        miscompiles the graph in that form — the output becomes essentially
        input-independent — so the model must be reshaped to a static input
        shape before compiling.  Compiled models are cached per shape.
        """
        key = ("moge", tuple(shape))
        if key not in self._compiled_models:
            path = self._model_paths.get("moge")
            if path is None:
                raise RuntimeError("OV model 'moge' is not registered.")
            model = self.core.read_model(path)
            model.reshape({model.inputs[0].get_any_name(): ov.PartialShape(list(shape))})
            self._log(f"  [compile] moge{tuple(shape)} → {self.device}...")
            self._compiled_models[key] = self.core.compile_model(model, self.device)
        return self._compiled_models[key]

    def _load_moge(self, moge_checkpoint_dir: Optional[str]):
        """Register MoGe OV IR model for pointmap computation.

        MoGe is now fully exported to OV IR (moge.xml).  The OV model is
        lazily compiled on first use via _get_model('moge'), just like every
        other model in the pipeline.
        """
        exported = self.cfg.get("exported", {})
        moge_path = self._resolve_path(exported.get("moge"))
        if moge_path and Path(moge_path).exists():
            self._log("  ✓ MoGe OV model registered")
        else:
            self._log(f"  ✗ MoGe OV model not found at {moge_path}. "
                      "Pointmap will be zero. Re-run export.py to create moge.xml.")

    def _load_embedder_weights(self):
        """Load EmbedderFuser positional embeddings, PointPatchEmbed outer weights,
        and SLAT sparse IO + t_embedder weights from pre-exported files.
        """
        try:
            exported = self.cfg.get("exported", {})

            # ── PointPatchEmbed outer: now a registered OV IR model ─────────────
            ppe_ov_path = self._resolve_path(exported.get("ss_pointpatch_outer"))
            if ppe_ov_path and Path(ppe_ov_path).exists():
                self._log("  ✓ PointPatchEmbed outer OV model registered")
                self.ppe_outer = None  # no longer using Python class
            else:
                # Fallback to legacy .npz weights (Python implementation)
                ppe_path = self._resolve_path(exported.get("ss_pointpatch_outer_weights"))
                if ppe_path and Path(ppe_path).exists():
                    npz = np.load(ppe_path)
                    self.ppe_outer = PointPatchOuterPreprocess(
                        point_proj_weight=npz["point_proj_weight"],
                        point_proj_bias=npz["point_proj_bias"],
                        invalid_xyz_token=npz["invalid_xyz_token"],
                        input_size=self.cfg.get("pointpatch_input_size", 256),
                    )
                    self._log("  ✓ PointPatchEmbed outer weights loaded (legacy .npz)")
                else:
                    self._log("  ✗ PointPatchEmbed outer not found")
                    self.ppe_outer = None

            # ── SS positional embeddings (idx_emb.npy) ────────────────────────
            ss_idx_path = self._resolve_path(exported.get("ss_idx_emb"))
            if ss_idx_path and Path(ss_idx_path).exists():
                ss_idx_emb = np.load(ss_idx_path)   # (3, 1024)
                self.ss_embedder_pos = EmbedderFuserPositionalEmbed(
                    idx_emb=ss_idx_emb,
                    positional_embed_map={"cropped": 0, "full": 1},
                )
                self._log(f"  ✓ SS idx_emb loaded {ss_idx_emb.shape}")
            else:
                self._log(f"  ✗ SS idx_emb not found: {ss_idx_path}")
                self.ss_embedder_pos = None

            # ── SLAT positional embeddings (idx_emb.npy) ─────────────────────
            slat_idx_path = self._resolve_path(exported.get("slat_idx_emb"))
            if slat_idx_path and Path(slat_idx_path).exists():
                slat_idx_emb = np.load(slat_idx_path)  # (3, 1024) — SLAT uses 2
                self.slat_embedder_pos = EmbedderFuserPositionalEmbed(
                    idx_emb=slat_idx_emb,
                    positional_embed_map={"cropped": 0, "full": 1},
                )
                self._log(f"  ✓ SLAT idx_emb loaded {slat_idx_emb.shape}")
            else:
                self._log(f"  ✗ SLAT idx_emb not found: {slat_idx_path}")
                self.slat_embedder_pos = None

            # ── SLAT sparse IO weights (from manifest) ────────────────────────
            manifest_path = self._resolve_path(exported.get("slat_sparse_io_manifest"))
            weights_dir   = self._resolve_path(exported.get("slat_sparse_io_weights_dir"))
            if manifest_path and Path(manifest_path).exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                self.slat_io_weights: Dict[str, np.ndarray] = {}
                for layer_key, meta in manifest["layers"].items():
                    fpath = Path(weights_dir) / meta["file"]
                    self.slat_io_weights[layer_key] = np.load(str(fpath)).astype(np.float32)
                self._log(f"  ✓ SLAT sparse IO weights loaded ({len(self.slat_io_weights)} tensors)")
            else:
                self._log("  ✗ SLAT sparse IO manifest not found")
                self.slat_io_weights = {}

            # SLAT t_embedder: uses OV model slat_t_embedder (no torch dependency needed)

            # Build per-layer OV SubMConv3d models (replaces Python fallback)
            if self.slat_io_weights:
                self._build_slat_io_conv_models()

        except Exception as e:
            import traceback
            self._log(f"  ✗ Failed to load embedder weights: {e}")
            if self.verbose:
                traceback.print_exc()
            self.ppe_outer         = None
            self.ss_embedder_pos   = None
            self.slat_embedder_pos = None
            self.slat_io_weights   = {}
            self._slat_conv_compiled = {}
            self._slat_conv_params   = {}
            self._slat_conv_model_paths = {}

    def _build_slat_io_conv_models(self):
        """Build per-SubMConv3d OV model parameter sets using SparseConv3dEngine extension.

        Stores XML + binary weight blobs for each of the 8 submanifold sparse
        convolutions in the SLAT IO blocks. Models are lazily compiled on first
        use via _run_subm_conv3d() to avoid OpenCL init conflicts at startup.
        """
        import tempfile

        LAYER_SUBM_CONV = 0  # must match C++ side

        def _make_subm_conv_xml(max_voxels, C_in, C_out, GS, n_params):
            """Generate OV IR XML for a single SubMConv3d (SparseConv3dEngine)."""
            layer_defs_str = f"{LAYER_SUBM_CONV},{C_in},{C_out},27"
            return f"""\
<?xml version="1.0"?>
<net name="sparse_conv_3d" version="11">
  <layers>
    <layer id="0" name="features" type="Parameter" version="opset1">
      <data shape="{max_voxels},{C_in}" element_type="f32"/>
      <output><port id="0" precision="FP32" names="features">
        <dim>{max_voxels}</dim><dim>{C_in}</dim>
      </port></output>
    </layer>
    <layer id="1" name="coords" type="Parameter" version="opset1">
      <data shape="{max_voxels},4" element_type="i32"/>
      <output><port id="0" precision="I32" names="coords">
        <dim>{max_voxels}</dim><dim>4</dim>
      </port></output>
    </layer>
    <layer id="2" name="num_voxels" type="Parameter" version="opset1">
      <data shape="1" element_type="i32"/>
      <output><port id="0" precision="I32" names="num_voxels">
        <dim>1</dim>
      </port></output>
    </layer>
    <layer id="3" name="params" type="Const" version="opset1">
      <data element_type="f32" shape="{n_params}" offset="0" size="{n_params * 4}"/>
      <output><port id="0" precision="FP32">
        <dim>{n_params}</dim>
      </port></output>
    </layer>
    <layer id="4" name="sparse_conv_3d" type="SparseConv3dEngine" version="sam3d">
      <data max_voxels="{max_voxels}" in_channels="{C_in}" out_channels="{C_out}"
            num_layers="1" spatial_resolution="{GS}" layer_defs="{layer_defs_str}"/>
      <input>
        <port id="0"><dim>{max_voxels}</dim><dim>{C_in}</dim></port>
        <port id="1"><dim>{max_voxels}</dim><dim>4</dim></port>
        <port id="2"><dim>1</dim></port>
        <port id="3"><dim>{n_params}</dim></port>
      </input>
      <output><port id="4" precision="FP32" names="out_features">
        <dim>{max_voxels}</dim><dim>{C_out}</dim>
      </port></output>
    </layer>
    <layer id="5" name="output" type="Result" version="opset1">
      <input><port id="0">
        <dim>{max_voxels}</dim><dim>{C_out}</dim>
      </port></input>
    </layer>
  </layers>
  <edges>
    <edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
    <edge from-layer="1" from-port="0" to-layer="4" to-port="1"/>
    <edge from-layer="2" from-port="0" to-layer="4" to-port="2"/>
    <edge from-layer="3" from-port="0" to-layer="4" to-port="3"/>
    <edge from-layer="4" from-port="4" to-layer="5" to-port="0"/>
  </edges>
</net>
"""

        MAX_N = 65536
        GS    = self.cfg.get("slat_grid_size", 64)
        W     = self.slat_io_weights
        # Store params for lazy compilation on first use
        self._slat_conv_compiled: Dict[str, Any] = {}
        self._slat_conv_params: Dict[str, Any] = {}
        self._slat_conv_model_paths: Dict[str, Tuple[str, str]] = {}

        # Map of (model_key → weight_key_prefix_in_slat_io_weights)
        conv_specs = [
            ("input_blocks.0.conv1", "input_blocks.0.conv1.conv"),
            ("input_blocks.0.conv2", "input_blocks.0.conv2.conv"),
            ("input_blocks.1.conv1", "input_blocks.1.conv1.conv"),
            ("input_blocks.1.conv2", "input_blocks.1.conv2.conv"),
            ("out_blocks.0.conv1",   "out_blocks.0.conv1.conv"),
            ("out_blocks.0.conv2",   "out_blocks.0.conv2.conv"),
            ("out_blocks.1.conv1",   "out_blocks.1.conv1.conv"),
            ("out_blocks.1.conv2",   "out_blocks.1.conv2.conv"),
        ]

        for model_key, wprefix in conv_specs:
            wk = f"{wprefix}.weight"
            bk = f"{wprefix}.bias"
            if wk not in W:
                self._log(f"  ✗ SLAT IO conv missing weight: {wk}")
                continue

            wt = W[wk]    # (C_out, 3, 3, 3, C_in)
            bt = W[bk]    # (C_out,)
            C_out, _, _, _, C_in = wt.shape

            # Pack: [weights_flat | scale=1 (identity BN) | bias]
            packed = np.concatenate([
                wt.reshape(-1).astype(np.float32),
                np.ones(C_out, dtype=np.float32),
                bt.astype(np.float32),
            ])
            n_params = packed.shape[0]

            xml_str = _make_subm_conv_xml(MAX_N, C_in, C_out, GS, n_params)

            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as bf:
                bf.write(packed.tobytes())
                bin_path = bf.name
            with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as xf:
                xf.write(xml_str)
                xml_path = xf.name

            # Store paths for lazy compilation on first use (avoids compile-time
            # OpenCL conflict during pipeline init)
            self._slat_conv_model_paths[model_key] = (xml_path, bin_path)
            # Store params for reference — n_params needed for dynamic MAX_N recompilation
            self._slat_conv_params[model_key] = {"C_in": C_in, "C_out": C_out, "n_params": n_params}
            self._log(f"  ✓ SLAT IO conv ready: {model_key} ({C_in}→{C_out})")

    def _run_subm_conv3d(self, model_key: str,
                         feats: np.ndarray,
                         coords_zyx: np.ndarray) -> np.ndarray:
        """Run a single submanifold sparse conv3d via the OV SparseConv3dEngine.

        Args:
            model_key:   Key into self._slat_conv_compiled (e.g. "input_blocks.0.conv1")
            feats:       (N, C_in) float32
            coords_zyx:  (N, 3) int32  — [z, y, x]

        Returns:
            (N, C_out) float32
        """
        N     = feats.shape[0]
        # Dynamic MAX_N: round up to next multiple of 256 for alignment.
        # This avoids allocating 65536×C_in arrays (up to 512 MB) when only
        # ~18K-26K voxels are active — reduces allocation by 2-3×.
        MAX_N = ((N + 255) // 256) * 256

        # Lazy compilation: compile OV model on first use (deferred from init to
        # avoid OpenCL conflict during pipeline initialization).
        # Recompile if the voxel count exceeds the current model's MAX_N.
        cur_max = getattr(self, '_slat_conv_max_n', {}).get(model_key, 0)
        if model_key not in self._slat_conv_compiled or MAX_N > cur_max:
            if model_key not in self._slat_conv_model_paths:
                raise RuntimeError(f"No compiled model or model path for '{model_key}'")
            xml_path, bin_path = self._slat_conv_model_paths[model_key]
            # Regenerate XML with actual MAX_N
            C_in_  = feats.shape[1]
            C_out_ = self._slat_conv_params[model_key]["C_out"]
            n_params_ = self._slat_conv_params.get(model_key, {}).get("n_params")
            if n_params_ is None:
                # Read from existing model
                model = self.core.read_model(xml_path, bin_path)
            else:
                LAYER_SUBM_CONV = 0
                GS = self.cfg.get("slat_grid_size", 64)
                layer_defs_str = f"{LAYER_SUBM_CONV},{C_in_},{C_out_},27"
                xml_str = f"""\
<?xml version="1.0"?>
<net name="sparse_conv_3d" version="11">
  <layers>
    <layer id="0" name="features" type="Parameter" version="opset1">
      <data shape="{MAX_N},{C_in_}" element_type="f32"/>
      <output><port id="0" precision="FP32" names="features">
        <dim>{MAX_N}</dim><dim>{C_in_}</dim>
      </port></output>
    </layer>
    <layer id="1" name="coords" type="Parameter" version="opset1">
      <data shape="{MAX_N},4" element_type="i32"/>
      <output><port id="0" precision="I32" names="coords">
        <dim>{MAX_N}</dim><dim>4</dim>
      </port></output>
    </layer>
    <layer id="2" name="num_voxels" type="Parameter" version="opset1">
      <data shape="1" element_type="i32"/>
      <output><port id="0" precision="I32" names="num_voxels">
        <dim>1</dim>
      </port></output>
    </layer>
    <layer id="3" name="params" type="Const" version="opset1">
      <data element_type="f32" shape="{n_params_}" offset="0" size="{n_params_ * 4}"/>
      <output><port id="0" precision="FP32">
        <dim>{n_params_}</dim>
      </port></output>
    </layer>
    <layer id="4" name="sparse_conv_3d" type="SparseConv3dEngine" version="sam3d">
      <data max_voxels="{MAX_N}" in_channels="{C_in_}" out_channels="{C_out_}"
            num_layers="1" spatial_resolution="{GS}" layer_defs="{layer_defs_str}"/>
      <input>
        <port id="0"><dim>{MAX_N}</dim><dim>{C_in_}</dim></port>
        <port id="1"><dim>{MAX_N}</dim><dim>4</dim></port>
        <port id="2"><dim>1</dim></port>
        <port id="3"><dim>{n_params_}</dim></port>
      </input>
      <output><port id="4" precision="FP32" names="out_features">
        <dim>{MAX_N}</dim><dim>{C_out_}</dim>
      </port></output>
    </layer>
    <layer id="5" name="output" type="Result" version="opset1">
      <input><port id="0">
        <dim>{MAX_N}</dim><dim>{C_out_}</dim>
      </port></input>
    </layer>
  </layers>
  <edges>
    <edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
    <edge from-layer="1" from-port="0" to-layer="4" to-port="1"/>
    <edge from-layer="2" from-port="0" to-layer="4" to-port="2"/>
    <edge from-layer="3" from-port="0" to-layer="4" to-port="3"/>
    <edge from-layer="4" from-port="4" to-layer="5" to-port="0"/>
  </edges>
</net>
"""
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as xf:
                    xf.write(xml_str)
                    new_xml = xf.name
                model = self.core.read_model(new_xml, bin_path)
            self._log(f"  [sparse conv compile] {model_key} (MAX_N={MAX_N})...")
            compiled = self.core.compile_model(model, "CPU")
            self._slat_conv_compiled[model_key] = compiled
            if not hasattr(self, '_slat_conv_max_n'):
                self._slat_conv_max_n = {}
            self._slat_conv_max_n[model_key] = MAX_N
            # Clear cached infer request on recompile
            if hasattr(self, '_conv_infer_reqs') and model_key in self._conv_infer_reqs:
                del self._conv_infer_reqs[model_key]

        compiled = self._slat_conv_compiled[model_key]
        C_in  = feats.shape[1]
        C_out = self._slat_conv_params[model_key]["C_out"]

        feat_pad  = np.zeros((MAX_N, C_in), dtype=np.float32)
        feat_pad[:N] = feats

        # Cache coord_pad — coordinates don't change across SLAT steps
        cache_key = "_conv_coord_pad"
        cached = getattr(self, cache_key, None)
        if cached is None or cached.shape[0] != MAX_N or cached.shape[1] != 4:
            coord_pad = np.zeros((MAX_N, 4), dtype=np.int32)
            coord_pad[:N] = np.column_stack([
                np.zeros(N, dtype=np.int32),   # batch = 0
                coords_zyx[:, 0].astype(np.int32),  # z → C++ slot 1
                coords_zyx[:, 1].astype(np.int32),  # y → C++ slot 2
                coords_zyx[:, 2].astype(np.int32),  # x → C++ slot 3
            ])
            setattr(self, cache_key, coord_pad)
        coord_pad = getattr(self, cache_key)
        num_vox = np.array([N], dtype=np.int32)

        # Reuse infer request — avoids per-call allocation
        if not hasattr(self, '_conv_infer_reqs'):
            self._conv_infer_reqs = {}
        if model_key not in self._conv_infer_reqs:
            self._conv_infer_reqs[model_key] = compiled.create_infer_request()
        req = self._conv_infer_reqs[model_key]
        req.set_input_tensor(0, ov.Tensor(feat_pad))
        req.set_input_tensor(1, ov.Tensor(coord_pad))
        req.set_input_tensor(2, ov.Tensor(num_vox))
        req.infer()
        return req.get_output_tensor(0).data[:N, :C_out].copy()

    # ─── SLAT t_embedder (replicated in Python for sparse IO emb_layers) ──────

    def _compute_slat_t_emb(self, t_scaled: float) -> np.ndarray:
        """Compute SLAT t_embedder + IO emb outputs via merged slat_t_emb_all model.

        Args:
            t_scaled: timestep already multiplied by time_scale (e.g., t * 1000.0)

        Returns:
            tuple: (t_emb, emb_in0, emb_in1, emb_out0, emb_out1)
        """
        t_arr = np.array([t_scaled], dtype=np.float32)
        out = self._ov_infer("slat_t_emb_all", {"t": t_arr})
        # Output keys may be named or integer — collect by position
        vals = list(out.values())
        return vals[0], vals[1], vals[2], vals[3], vals[4]  # t_emb, emb_in0, emb_in1, emb_out0, emb_out1

    # ─── Inference helpers ────────────────────────────────────────────────────

    def _ov_infer(self, model_name: str, inputs: Dict[str, np.ndarray]
                  ) -> Dict[str, np.ndarray]:
        """Run an OV compiled model and return output dict."""
        m = self._get_model(model_name)
        result = m(inputs)
        # Return as dict keyed by output tensor names (fallback to numeric index)
        outputs = {}
        for i, out in enumerate(m.outputs):
            names = out.get_tensor().get_names()
            key = next(iter(names)) if names else i
            outputs[key] = result[i]
        return outputs

    def _get_attn_request(self, key="cond"):
        """Get or create a pre-allocated InferRequest for the attention model."""
        if not hasattr(self, "_attn_req_cache"):
            self._attn_req_cache = {}
        if key not in self._attn_req_cache:
            self._attn_req_cache[key] = \
                self._get_model("slat_attn_blocks").create_infer_request()
        return self._attn_req_cache[key]

    def _run_dino(self, model_name: str, img_nchw: np.ndarray) -> np.ndarray:
        """Run DINOv2 OV model. Returns (1, N_tokens, D).

        If the model was exported with 3ch input (slat_dino_*) and input is 1ch,
        replicate to 3ch. ss_dino_mask was exported with 1ch input and handles
        expansion internally.
        """
        x = img_nchw.astype(np.float32)
        # SLAT DINOs were exported with 3ch example — expand 1ch masks
        if x.shape[1] == 1 and model_name.startswith("slat_dino"):
            x = np.repeat(x, 3, axis=1)
        out = self._ov_infer(model_name, {"x": x})
        return out[0]  # single output, keyed by index 0

    def _run_proj_net(self, model_name: str, tokens: np.ndarray) -> np.ndarray:
        """Run EmbedderFuser projection net. Returns (1, N, D_out)."""
        out = self._ov_infer(model_name, {"x": tokens.astype(np.float32)})
        return out[0]  # single output

    # ─── Condition embedding ──────────────────────────────────────────────────

    def _get_pointmap(self, rgb_hwc: np.ndarray) -> Optional[np.ndarray]:
        """Run MoGe OV model to get pointmap (3, H, W) in PyTorch3D camera space."""
        if "moge" not in self._model_paths:
            return None
        try:
            # MoGe OV model expects (1, 3, H, W) float32 in [0, 1].  Geometry
            # depends on the aspect ratio, not the pixel count, so the long side
            # may be capped to MOGE_MAX_SIDE with the aspect ratio preserved.
            H0, W0 = rgb_hwc.shape[:2]
            max_side = int(os.environ.get("SAM3D_MOGE_MAX_SIDE", MOGE_MAX_SIDE))
            src = rgb_hwc
            if max_side > 0 and max(H0, W0) > max_side:
                scale = max_side / max(H0, W0)
                Ht, Wt = max(1, round(H0 * scale)), max(1, round(W0 * scale))
                ys = (np.arange(Ht) * H0 / Ht).astype(int)
                xs = (np.arange(Wt) * W0 / Wt).astype(int)
                src = rgb_hwc[ys[:, None], xs[None, :], :]
            img_nchw = src.transpose(2, 0, 1)[None].astype(np.float32)  # (1,3,H,W)
            if img_nchw.max() > 1.01:
                img_nchw = img_nchw / 255.0

            # Compile for this exact shape — a dynamic-shape MoGe is miscompiled
            # on the GPU plugin (see _get_moge_model).
            compiled = self._get_moge_model(img_nchw.shape)
            result = compiled({"image": img_nchw})
            ov_out = {}
            for i, out in enumerate(compiled.outputs):
                names = out.get_tensor().get_names()
                ov_out[next(iter(names)) if names else i] = result[i]
            # MoGe outputs are named: "points" and "mask"
            key0 = "points" if "points" in ov_out else 0
            points_bhw3 = ov_out[key0]  # (1, S, S, 3)  — affine camera-space point map
            key1 = "mask" if "mask" in ov_out else 1
            mask_bhw = ov_out[key1]     # (1, S, S) — validity logits

            # The exported IR wraps MoGeModel.forward(), which returns the raw
            # AFFINE-INVARIANT point map (z defined only up to an unknown shift).
            # The baseline calls MoGeModel.infer(force_projection=False), which
            # also recovers that shift and masks invalid pixels — reproduce that
            # post-processing here (it runs on a 64x64 downsample, so it is cheap).
            pts = points_bhw3.astype(np.float32)                 # (1, S, S, 3)
            mask_binary = mask_bhw.astype(np.float32) > MOGE_MASK_THRESHOLD  # (1, S, S)
            shift = _moge_recover_shift(pts, mask_binary)        # (1,)
            pts = pts + np.stack(
                [np.zeros_like(shift), np.zeros_like(shift), shift], axis=-1
            )[:, None, None, :]
            # infer(apply_mask=True) marks invalid pixels as inf; ObjectCentricSSI
            # and PointPatchEmbed both treat non-finite entries as invalid.
            pts = np.where(mask_binary[..., None], pts, np.float32(np.inf))

            # Convert from MoGe convention (OpenCV: x right, y down, z forward)
            # to PyTorch3D convention via camera_to_pytorch3d_camera rotation:
            #   look_at_view_transform(eye=[0,0,-1], up=[0,-1,0]) → flip X and Y, keep Z.
            pointmap = pts[0].copy()   # (S, S, 3)
            pointmap[..., 0] = -pointmap[..., 0]
            pointmap[..., 1] = -pointmap[..., 1]

            # Resize the point map back to the RGB resolution the rest of the
            # pipeline works in (the baseline's resize_all_to_same_size does the
            # same before any cropping).
            Hp, Wp = pointmap.shape[:2]
            if (Hp, Wp) != (H0, W0):
                ys_r = (np.arange(H0) * Hp / H0).astype(int)
                xs_r = (np.arange(W0) * Wp / W0).astype(int)
                pointmap = pointmap[ys_r[:, None], xs_r[None, :], :]
            return pointmap.transpose(2, 0, 1)  # (3, H, W)

        except Exception as e:
            self._log(f"  MoGe OV inference failed: {e}")
            return None

    def _embed_conditions_ss(
        self,
        image_crop: np.ndarray,   # (3, 518, 518) normalized
        mask_crop: np.ndarray,    # (1, 518, 518) binary
        image_full: np.ndarray,   # (3, 518, 518) normalized
        mask_full: np.ndarray,    # (1, 518, 518) binary
        pointmap_crop: Optional[np.ndarray] = None,  # (3, 518, 518) or None
        pointmap_full: Optional[np.ndarray] = None,  # (3, 518, 518) or None
    ) -> np.ndarray:
        """Run SS EmbedderFuser to produce condition tokens (1, N_cond, 1024).

        Implements EmbedderFuser.forward() in Python+OV steps:
        1. Each sub-embedder processes its kwargs
        2. Projection nets applied per sub-embedder
        3. Positional embeddings added
        4. Concat all tokens

        Returns: (1, SS_COND_NUM_TOKENS, 1024) where SS_COND_NUM_TOKENS=7544
        """
        all_tokens = []

        def _process_dino_kwarg(model_key: str, img_nchw: np.ndarray,
                                 proj_idx: int, pos_group: str) -> np.ndarray:
            """Run DINOv2 + projection net + pos embed → (1, N_dino, 1024)."""
            tokens = self._run_dino(model_key, img_nchw)          # (1, 1374, D_dino)
            tokens = self._run_proj_net(f"ss_embedder_proj_{proj_idx}", tokens)  # (1, 1374, 1024)
            if self.ss_embedder_pos:
                tokens = self.ss_embedder_pos.apply(tokens, pos_group)
            return tokens

        # Sub-embedder 0 (DINO image + proj_0):
        all_tokens.append(_process_dino_kwarg("ss_dino_image",
                                               image_crop[None], 0, "cropped"))
        all_tokens.append(_process_dino_kwarg("ss_dino_image",
                                               image_full[None], 0, "full"))

        # Sub-embedder 1 (DINO mask + proj_1):
        all_tokens.append(_process_dino_kwarg("ss_dino_mask",
                                               mask_crop[None], 1, "cropped"))
        all_tokens.append(_process_dino_kwarg("ss_dino_mask",
                                               mask_full[None], 1, "full"))

        # Sub-embedder 2 (PointPatchEmbed): run real pointmap conditioning.
        D = self.cfg.get("cond_embed_dim", 1024)
        N_pp = self.cfg.get("pointpatch_num_tokens", 1024)
        have_pp = ("ss_pointpatch" in self._model_paths
                   and pointmap_crop is not None and pointmap_full is not None)
        if have_pp:
            all_tokens.append(self._run_pointpatch(pointmap_crop, "cropped"))
            all_tokens.append(self._run_pointpatch(pointmap_full, "full"))
        else:
            # Fallback: pointmap dropped → zeros (degrades depth; only if model missing)
            for _ in range(2):
                all_tokens.append(np.zeros((1, N_pp, D), dtype=np.float32))

        # Concatenate along token dimension
        return np.concatenate(all_tokens, axis=1)  # (1, 7544, 1024)

    def _run_pointpatch(self, pointmap_518: np.ndarray, pos_group: str) -> np.ndarray:
        """Run full PointPatchEmbed → projection net #2 → positional embedding.

        Args:
            pointmap_518: (3, S, S) SSI-normalized pointmap (NaN = invalid pixel).
            pos_group: 'cropped' or 'full'.
        Returns: (1, 1024, 1024) condition tokens.
        """
        valid = np.isfinite(pointmap_518).all(axis=0, keepdims=True).astype(np.float32)  # (1,S,S)
        xyz = np.nan_to_num(pointmap_518, nan=0.0).astype(np.float32)[None]               # (1,3,S,S)
        valid = valid[None]                                                                # (1,1,S,S)
        out = self._ov_infer("ss_pointpatch", {"xyz": xyz, "valid": valid})
        tokens = out[0]                                             # (1, 1024, 512)
        tokens = self._run_proj_net("ss_embedder_proj_2", tokens)  # (1, 1024, 1024)
        if self.ss_embedder_pos:
            tokens = self.ss_embedder_pos.apply(tokens, pos_group)
        return tokens


    def _embed_conditions_slat(
        self,
        image_full: np.ndarray,  # (3, 518, 518) normalized
        mask_full: np.ndarray,   # (1, 518, 518) binary
        image_crop: np.ndarray,  # (3, 518, 518) normalized
        mask_crop: np.ndarray,   # (1, 518, 518) binary
    ) -> np.ndarray:
        """Run SLAT EmbedderFuser — DINO only, no PointPatchEmbed.

        Returns: (1, SLAT_COND_NUM_TOKENS, 1024)
        """
        all_tokens = []

        def _proc(model_key: str, img_nchw: np.ndarray,
                  proj_idx: int, pos_group: str) -> np.ndarray:
            tokens = self._run_dino(model_key, img_nchw)
            tokens = self._run_proj_net(f"slat_embedder_proj_{proj_idx}", tokens)
            if self.slat_embedder_pos:
                tokens = self.slat_embedder_pos.apply(tokens, pos_group)
            return tokens

        # slat_dino_0 = RGB (prenorm), slat_dino_1 = mask (prenorm, 1ch)
        all_tokens.append(_proc("slat_dino_0", image_crop[None], 0, "cropped"))
        all_tokens.append(_proc("slat_dino_0", image_full[None], 0, "full"))
        all_tokens.append(_proc("slat_dino_1", mask_crop[None], 1, "cropped"))
        all_tokens.append(_proc("slat_dino_1", mask_full[None], 1, "full"))

        return np.concatenate(all_tokens, axis=1)  # (1, 5496, 1024)

    # ─── SS Sampling (ShortCut 2-step ODE) ───────────────────────────────────

    def _ss_project_inputs(self, latents: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply all 5 input projections via merged ss_latent_in_all OV model."""
        ov_out = self._ov_infer("ss_latent_in_all", {
            "x_shape":  latents["shape"].astype(np.float32),
            "x_rot":    latents["6drotation_normalized"].astype(np.float32),
            "x_trans":  latents["translation"].astype(np.float32),
            "x_scale":  latents["scale"].astype(np.float32),
            "x_tscale": latents["translation_scale"].astype(np.float32),
        })
        vals = list(ov_out.values())
        return {
            "shape": vals[0],
            "6drotation_normalized": vals[1],
            "translation": vals[2],
            "scale": vals[3],
            "translation_scale": vals[4],
        }

    def _ss_project_outputs(self, h: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply all 5 output projections via merged ss_latent_out_all OV model."""
        ov_out = self._ov_infer("ss_latent_out_all", {
            "h_shape":  h["shape"].astype(np.float32),
            "h_rot":    h["6drotation_normalized"].astype(np.float32),
            "h_trans":  h["translation"].astype(np.float32),
            "h_scale":  h["scale"].astype(np.float32),
            "h_tscale": h["translation_scale"].astype(np.float32),
        })
        vals = list(ov_out.values())
        return {
            "shape": vals[0],
            "6drotation_normalized": vals[1],
            "translation": vals[2],
            "scale": vals[3],
            "translation_scale": vals[4],
        }

    def _merge_latent_share_transformer(
        self, h: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Merge pose modalities into a single token group.

        latent_share_transformer config:
          {"6drotation_normalized": ["6drotation_normalized", "translation",
                                     "scale", "translation_scale"]}
        → Concatenate 4 pose modalities (each (B, 1, 1024)) → (B, 4, 1024)

        Returns {"shape": ..., "6drotation_normalized": (B, 4, 1024)}
        """
        merged = {}
        merged["shape"] = h["shape"]
        # Concat rotation, translation, scale, translation_scale on token dim
        pose = np.concatenate([
            h["6drotation_normalized"],   # (B, 1, 1024)
            h["translation"],             # (B, 1, 1024)
            h["scale"],                   # (B, 1, 1024)
            h["translation_scale"],       # (B, 1, 1024)
        ], axis=1)                         # (B, 4, 1024)
        merged["6drotation_normalized"] = pose
        return merged

    def _split_latent_share_transformer(
        self, h_merged: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Inverse of merge: split (B, 4, 1024) back into the 4 modalities."""
        split = {}
        split["shape"] = h_merged["shape"]
        pose = h_merged["6drotation_normalized"]  # (B, 4, 1024)
        split["6drotation_normalized"] = pose[:, 0:1, :]  # (B, 1, 1024)
        split["translation"]           = pose[:, 1:2, :]
        split["scale"]                 = pose[:, 2:3, :]
        split["translation_scale"]     = pose[:, 3:4, :]
        return split

    def _sample_ss(self, cond: np.ndarray, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        """Sample sparse structure using ShortCut 25-step ODE (no_shortcut=True → d=0).

        Returns Dict with shape + pose latents at noise level t=1 (fully denoised).
        """
        cfg = self.cfg
        B = 1
        ss_tokens = cfg.get("ss_num_tokens", 4096)    # 4096 = 16^3
        ss_steps  = cfg.get("ss_inference_steps", 25)
        cfg_strength = cfg.get("ss_cfg_strength", 7.0)
        cfg_interval = cfg.get("ss_cfg_interval", [0, 500])
        time_scale   = cfg.get("ss_time_scale", 1000.0)
        rescale_t    = cfg.get("ss_rescale_t", 3.0)

        # ShortCut schedule with time warping (rescale_t=3 matches baseline).
        # d=0 because baseline always uses no_shortcut=True.
        ts, ds = shortcut_schedule(ss_steps, rescale_t)

        # Initialize latent noise for each modality
        noise: Dict[str, np.ndarray] = {
            "shape":                  rng.standard_normal((B, ss_tokens, 8)).astype(np.float32),
            "6drotation_normalized":  rng.standard_normal((B, 1, 6)).astype(np.float32),
            "translation":            rng.standard_normal((B, 1, 3)).astype(np.float32),
            "scale":                  rng.standard_normal((B, 1, 3)).astype(np.float32),
            "translation_scale":      rng.standard_normal((B, 1, 1)).astype(np.float32),
        }

        # For CFG the unconditional condition is all zeros (force_zeros_cond=True).
        # A zero cond makes every cross-attention key/value identical, so the
        # backbone output is bit-identical for any cond length — feed one token
        # instead of the full ~7.5k to skip the redundant work.
        cond_zeros = np.zeros((cond.shape[0], 1, cond.shape[2]), dtype=cond.dtype)

        x = noise

        for step_idx in range(ss_steps):
            t_cur = float(ts[step_idx])
            t_next = float(ts[step_idx + 1])
            dt = t_next - t_cur  # warped time difference for Euler update
            d = float(ds[step_idx])  # 0.0 (no_shortcut=True)
            t_scaled = t_cur * time_scale
            self._log(f"    SS step {step_idx+1}/{ss_steps}: t={t_cur:.4f}, dt={dt:.4f}, d={d:.4f}")

            # Apply input projections (per modality)
            h = self._ss_project_inputs(x)

            # Merge pose group
            h_merged = self._merge_latent_share_transformer(h)

            t_arr_1 = np.full((1,), t_scaled, dtype=np.float32)
            d_arr_1 = np.full((1,), d * time_scale, dtype=np.float32)  # always 0.0

            # CFG: only apply when t_scaled is within cfg_interval (baseline behavior)
            use_cfg = (cfg_interval[0] <= t_scaled <= cfg_interval[1]) and cfg_strength != 0.0
            if use_cfg:
                def _run_bb(cond_1b):
                    ov = self._ov_infer("ss_backbone", {
                        "shape": h_merged["shape"].astype(np.float32),
                        "pose":  h_merged["6drotation_normalized"].astype(np.float32),
                        "t":     t_arr_1,
                        "d":     d_arr_1,
                        "cond":  cond_1b.astype(np.float32),
                    })
                    return ov[0], ov[1]
                v_shape_cond, v_pose_cond     = _run_bb(cond)
                v_shape_uncond, v_pose_uncond = _run_bb(cond_zeros)
                # Baseline CFG formula: (1 + strength) * v_cond - strength * v_uncond
                v_merged = {
                    "shape": (1.0 + cfg_strength) * v_shape_cond - cfg_strength * v_shape_uncond,
                    "6drotation_normalized": (1.0 + cfg_strength) * v_pose_cond - cfg_strength * v_pose_uncond,
                }
                self._log(f"      CFG applied (t_scaled={t_scaled:.1f})")
            else:
                ov = self._ov_infer("ss_backbone", {
                    "shape": h_merged["shape"].astype(np.float32),
                    "pose":  h_merged["6drotation_normalized"].astype(np.float32),
                    "t":     t_arr_1,
                    "d":     d_arr_1,
                    "cond":  cond.astype(np.float32),
                })
                v_merged = {"shape": ov[0], "6drotation_normalized": ov[1]}
                self._log(f"      CFG skipped (t_scaled={t_scaled:.1f})")

            # Apply output projections to get velocity in latent space
            v_split = self._split_latent_share_transformer(v_merged)
            v_out = self._ss_project_outputs(v_split)

            # Euler update: x_{n+1} = x_n + dt * v  (dt is warped time difference)
            for key in x:
                x[key] = x[key] + dt * v_out[key]

        return x

    # ─── SS Decode ────────────────────────────────────────────────────────────

    def _decode_ss(self, shape_latent: np.ndarray) -> np.ndarray:
        """Run shape latent through SS decoder.

        Args:
            shape_latent: (1, 4096, 8) — shape modality output from SS sampling

        Returns:
            occupancy: (1, 1, 64, 64, 64) — occupancy logits
        """
        ss_spatial = self.cfg.get("ss_spatial_res", 16)
        ss_lat_ch  = self.cfg.get("ss_latent_channels", 8)

        # Reshape: (1, 4096, 8) → (1, 8, 16, 16, 16)
        latent_5d = shape_latent.transpose(0, 2, 1).reshape(
            1, ss_lat_ch, ss_spatial, ss_spatial, ss_spatial)

        out = self._ov_infer("ss_decoder", {"x": latent_5d.astype(np.float32)})
        return out[0]  # (1, 1, 64, 64, 64)

    # ─── SLAT Sampling (FlowMatching 12-step, no CFG) ────────────────────────

    def _to_f16_parallel(self, arr: np.ndarray) -> np.ndarray:
        """Convert a large float array to contiguous float16 across the thread
        pool. numpy's astype releases the GIL, so splitting the row range over
        workers turns a ~150ms single-threaded cast (68M elems) into ~20ms."""
        a = np.ascontiguousarray(arr)
        n = a.shape[0]
        pool = getattr(self, "_updown_pool", None)
        # Small arrays: not worth the dispatch overhead.
        if pool is None or n < 8192:
            return a.astype(np.float16)
        out = np.empty(a.shape, np.float16)
        workers = max(1, getattr(self, "_updown_workers", 8))
        chunk = (n + workers - 1) // workers

        def _cast(lo, hi):
            np.copyto(out[lo:hi], a[lo:hi].astype(np.float16))

        futs = []
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            futs.append(pool.submit(_cast, lo, hi))
        for f in futs:
            f.result()
        return out

    def _slat_gemm_gather_idx(self, coords: np.ndarray, N: int) -> np.ndarray:
        """Vectorized replica of build_subm_neighbor_map -> flat gather indices.

        coords: (N,3) int32 [x,y,z] (== [c0,c1,c2] the pipeline passes; extension
        reads x=c0,y=c1,z=c2). Returns [N*27] int32 with absent neighbors == N
        (the zero-pad row used by the GEMM resblock graph). k-ordering matches the
        extension exactly: idx=0; for dx: for dy: for dz: nb=find(x+dx,y+dy,z+dz).
        Cached per (N, coord-signature) since coords are constant across ODE steps.
        """
        if not hasattr(self, "_gemm_gidx_cache"):
            self._gemm_gidx_cache = {}
        x = coords[:, 0].astype(np.int64)
        y = coords[:, 1].astype(np.int64)
        z = coords[:, 2].astype(np.int64)
        key = (x << 20) | (y << 10) | z
        sig = (int(N), int(key[0]), int(key[-1]), int(key.sum()))
        cached = self._gemm_gidx_cache.get(sig)
        if cached is not None:
            return cached
        order = np.argsort(key, kind="stable")
        skeys = key[order]
        gidx = np.empty((N, 27), np.int64)
        kk = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nkey = ((x + dx) << 20) | ((y + dy) << 10) | (z + dz)
                    pos = np.searchsorted(skeys, nkey)
                    pos_c = np.clip(pos, 0, N - 1)
                    valid = (pos < N) & (skeys[pos_c] == nkey)
                    gidx[:, kk] = np.where(valid, order[pos_c], -1)
                    kk += 1
        flat = gidx.reshape(-1).astype(np.int32)
        flat[flat < 0] = N
        self._gemm_gidx_cache[sig] = flat
        return flat

    def _build_gemm_resblock(self, prefix: str, N: int, gidx: np.ndarray):
        """Build+compile an OV GEMM SparseResBlock3d (gather->MatMul on XMX).

        Reformulates each SubMConv3d as gather->reshape[N,27*C_in]->MatMul, which
        oneDNN routes to the GPU XMX matrix engine (~4x faster than the OpenCL
        vector kernels). Numerically validated cos>=0.99999 vs the extension
        (test_scripts/validate_gemm_resblock.py). Weight layout is a pure
        reinterpret conv_w.reshape(27, C_in, C_out) == the extension's own layout.

        The gather indices (gidx) are constant across all 25 ODE steps, so they
        are baked into the graph as a Constant (no per-call upload). Features are
        consumed as FP16 to halve the host->device DMA (the dominant cost for the
        2048-channel out_blocks.0 input).
        """
        cache_key = (prefix, N)
        if not hasattr(self, "_gemm_rb_cache"):
            self._gemm_rb_cache = {}
        if cache_key in self._gemm_rb_cache:
            return self._gemm_rb_cache[cache_key]

        W = self.slat_io_weights
        ops = ov.opset10
        T = ov.Type

        conv1_w = W.get(f"{prefix}.conv1.conv.weight").astype(np.float32)  # (C_mid,3,3,3,C_in)
        conv1_b = W.get(f"{prefix}.conv1.conv.bias").astype(np.float32)
        conv2_w = W.get(f"{prefix}.conv2.conv.weight").astype(np.float32)  # (C_out,3,3,3,C_mid)
        conv2_b = W.get(f"{prefix}.conv2.conv.bias").astype(np.float32)
        norm1_w = W.get(f"{prefix}.norm1.weight").astype(np.float32)
        norm1_b = W.get(f"{prefix}.norm1.bias").astype(np.float32)
        skip_w = W.get(f"{prefix}.skip_connection.weight")
        has_skip = skip_w is not None

        C_mid = conv1_w.shape[0]
        C_in = conv1_w.shape[-1]
        C_out = conv2_w.shape[0]
        W1 = conv1_w.reshape(27, C_in, C_mid).reshape(27 * C_in, C_mid)
        W2 = conv2_w.reshape(27, C_mid, C_out).reshape(27 * C_mid, C_out)

        def cf16(a):
            a = np.ascontiguousarray(a.astype(np.float16))
            return ov.op.Constant(T.f16, ov.Shape(list(a.shape)), a.flatten())

        # Baked gather-index Constant (constant across all ODE steps).
        gidx_flat = np.ascontiguousarray(gidx.astype(np.int32))
        gidx_c = ov.op.Constant(T.i32, ov.Shape([N * 27]), gidx_flat)

        def silu(v):
            return ops.multiply(v, ops.sigmoid(v))

        def mvn(v):
            # MVN must run in f32: the variance of large intermediate feats
            # (values up to ~3e4) overflows f16 and yields NaN.
            ax = ov.op.Constant(T.i64, ov.Shape([1]), [1])
            v32 = ops.convert(v, T.f32)
            m = ops.mvn(v32, ax, True, 1e-6, "inside_sqrt")
            return ops.convert(m, T.f16)

        def gmm(h, ci, w):
            zrow = ov.op.Constant(T.f16, ov.Shape([1, ci]), np.zeros(ci, np.float16))
            hp = ops.concat([h, zrow], 0)                       # [N+1, ci]
            ax0 = ov.op.Constant(T.i32, ov.Shape([]), [0])
            g = ops.gather(hp, gidx_c, ax0)                     # [N*27, ci]
            rs = ov.op.Constant(T.i64, ov.Shape([2]), [N, 27 * ci])
            A = ops.reshape(g, rs, False)                       # [N, 27*ci]
            return ops.matmul(A, cf16(w), False, False)         # [N, out_ch]

        feats16 = ov.op.Parameter(T.f16, ov.Shape([N, C_in]))
        scale_p = ov.op.Parameter(T.f32, ov.Shape([1, C_mid]))
        shift_p = ov.op.Parameter(T.f32, ov.Shape([1, C_mid]))
        scale16 = ops.convert(scale_p, T.f16)
        shift16 = ops.convert(shift_p, T.f16)

        # norm1 (affine) + silu
        n1 = mvn(feats16)
        n1 = ops.add(ops.multiply(n1, cf16(norm1_w.reshape(1, C_in))),
                     cf16(norm1_b.reshape(1, C_in)))
        h = silu(n1)
        # conv1 + bias
        h = ops.add(gmm(h, C_in, W1), cf16(conv1_b.reshape(1, C_mid)))
        # norm2 (non-affine) * (1+scale) + shift + silu
        n2 = mvn(h)
        one = ov.op.Constant(T.f16, ov.Shape([1, C_mid]), np.ones(C_mid, np.float16))
        n2 = ops.add(ops.multiply(n2, ops.add(one, scale16)), shift16)
        h = silu(n2)
        # conv2 + bias
        h = ops.add(gmm(h, C_mid, W2), cf16(conv2_b.reshape(1, C_out)))
        # skip + residual
        if has_skip:
            skip_b = W.get(f"{prefix}.skip_connection.bias").astype(np.float32)
            sk = ops.matmul(feats16, cf16(np.ascontiguousarray(skip_w.astype(np.float32).T)),
                            False, False)
            sk = ops.add(sk, cf16(skip_b.reshape(1, C_out)))
            out16 = ops.add(h, sk)
        else:
            out16 = ops.add(h, feats16)
        out = ops.convert(out16, T.f32)

        model = ov.Model([out], [feats16, scale_p, shift_p], f"gemm_rb_{prefix}")
        cm = self.core.compile_model(model, self.device,
                                     {"INFERENCE_PRECISION_HINT": "f16"})
        req = cm.create_infer_request()
        self._log(f"  [gemm_resblock] compiled {prefix} N={N} "
                  f"C_in={C_in} C_mid={C_mid} C_out={C_out} skip={has_skip} (f16 in, gidx baked)")
        # Pre-allocate the input tensors once and keep numpy views onto the
        # OV-managed memory, so per-call ov.Tensor allocation is avoided.
        req.set_input_tensor(0, ov.Tensor(np.zeros((N, C_in), dtype=np.float16)))
        req.set_input_tensor(1, ov.Tensor(np.zeros((1, C_mid), dtype=np.float32)))
        req.set_input_tensor(2, ov.Tensor(np.zeros((1, C_mid), dtype=np.float32)))
        feats_buf = np.array(req.get_input_tensor(0).data, copy=False)
        scale_buf = np.array(req.get_input_tensor(1).data, copy=False)
        shift_buf = np.array(req.get_input_tensor(2).data, copy=False)
        entry = (req, C_mid, C_out, feats_buf, scale_buf, shift_buf)
        self._gemm_rb_cache[cache_key] = entry
        return entry

    def _slat_sparse_resblock(
        self,
        feats: np.ndarray,       # (N, C_in)
        coords: np.ndarray,      # (N, 3) int32
        emb_out: np.ndarray,     # (1, 2*C_out) — pre-computed from slat_t_emb_all
        prefix: str,             # e.g. "input_blocks.0"
        grid_size: int = 64,
    ) -> np.ndarray:
        """SparseResBlock3d using OV models for learned ops and OV extension for conv3d.

        Architecture (from structured_latent_flow.py SparseResBlock3d):
          scale, shift = chunk(emb_out, 2)                — pre-computed by caller
          h = norm1(feats) (affine) → silu                — OV slat_io_{prefix}_norm1
          h = conv1(h)                                    — OV SparseConv3dEngine
          h = norm2(h) (no affine) * (1+scale) + shift → silu  — element-wise math
          h = conv2(h)                                    — OV SparseConv3dEngine
          skip = skip_connection(feats) if ch differ      — OV slat_io_{prefix}_skip
          return skip + h
        """
        W = self.slat_io_weights
        safe_name = prefix.replace(".", "_")

        # ── GEMM-on-XMX path: whole resblock as an OV gather->MatMul graph ──
        # Routes SubMConv3d to the GPU matrix engine instead of the OpenCL vector
        # kernels.  Default; set SAM3D_GEMM_CONV=0 to use the extension path.
        if getattr(self, "_use_gemm_conv", None) is None:
            self._use_gemm_conv = os.environ.get("SAM3D_GEMM_CONV", "1") != "0"
        fused_key = f"slat_io_{safe_name}_fused_resblock"
        if self._use_gemm_conv and fused_key in self._model_paths:
            N = feats.shape[0]
            gidx = self._slat_gemm_gather_idx(coords[:, :3].astype(np.int32), N)
            req, C_mid, C_out, feats_buf, scale_buf, shift_buf = \
                self._build_gemm_resblock(prefix, N, gidx)
            # Write straight into the OV-managed input buffers. The feats copy
            # also performs the f32->f16 cast, so no separate conversion pass.
            feats_buf[:] = feats
            scale_buf[:] = emb_out[:, :C_mid].reshape(1, C_mid)
            shift_buf[:] = emb_out[:, C_mid:].reshape(1, C_mid)
            req.infer()
            out = np.array(req.get_output_tensor(0).data, copy=True)
            return out

        # ── Fast path: GPU-fused resblock (entire block in one extension call) ──
        fused_key = f"slat_io_{safe_name}_fused_resblock"
        if fused_key in self._model_paths:
            # emb_out is pre-computed by caller from slat_t_emb_all
            conv1_w = W.get(f"{prefix}.conv1.conv.weight")
            C_out = conv1_w.shape[0]
            scale = emb_out[0, :C_out].astype(np.float32)
            shift = emb_out[0, C_out:].astype(np.float32)

            N = feats.shape[0]
            coords_4d = np.column_stack([np.zeros(N, dtype=np.int32), coords[:, :3].astype(np.int32)])

            # ── Tight-shape fast path: dynamic model + pre-allocated buffer ──
            # Lazy-compile a dynamic-shaped version of the model for exact N.
            # This avoids 65536-row padding (~38ms wasted per call for 2048ch models).
            tight_key = (fused_key, N)
            if not hasattr(self, '_tight_resblock_cache'):
                self._tight_resblock_cache = {}
            if tight_key not in self._tight_resblock_cache:
                m_raw = self.core.read_model(self._model_paths[fused_key])
                C_in = feats.shape[1]
                m_raw.reshape({
                    'features':   ov.PartialShape([N, C_in]),
                    'coords':     ov.PartialShape([N, 4]),
                    'num_voxels': ov.PartialShape([1]),
                    'emb_scale':  ov.PartialShape([C_out]),
                    'emb_shift':  ov.PartialShape([C_out]),
                })
                cm = self.core.compile_model(m_raw, 'CPU')
                req = cm.create_infer_request()
                # Pre-set static inputs (coords, num_voxels) once — coords are constant per image
                req.set_input_tensor(1, ov.Tensor(coords_4d))
                req.set_input_tensor(2, ov.Tensor(np.array([N], dtype=np.int32)))
                # Pre-open mutable buffer views for feats, scale, shift
                # (set initial tensors so buffers are allocated)
                req.set_input_tensor(0, ov.Tensor(feats.astype(np.float32)))
                req.set_input_tensor(3, ov.Tensor(scale))
                req.set_input_tensor(4, ov.Tensor(shift))
                feat_buf = np.array(req.get_input_tensor(0).data, copy=False)
                scl_buf  = np.array(req.get_input_tensor(3).data, copy=False)
                sft_buf  = np.array(req.get_input_tensor(4).data, copy=False)
                self._log(f"  [tight_resblock] compiled {fused_key} N={N}")
                self._tight_resblock_cache[tight_key] = (req, feat_buf, scl_buf, sft_buf)

            req, feat_buf, scl_buf, sft_buf = self._tight_resblock_cache[tight_key]
            # Direct write into pre-allocated buffers — zero copy.
            # Skip feat copy if caller already wrote into feat_buf (same array).
            if feats is not feat_buf:
                feat_buf[:] = feats
            scl_buf[:]  = scale
            sft_buf[:]  = shift
            req.infer()
            # Output tensor is now exactly [N, C_out] — no slicing needed
            out = np.array(req.get_output_tensor(0).data, copy=True)
            return out

        # Determine C_out from conv1 weight shape
        conv1_w = W.get(f"{prefix}.conv1.conv.weight")
        C_out = conv1_w.shape[0]

        # emb_out is pre-computed by caller from slat_t_emb_all
        scale = emb_out[:, :C_out]
        shift = emb_out[:, C_out:]

        # ── norm1 + SiLU via OV ──
        norm1_model = f"slat_io_{safe_name}_norm1"
        if norm1_model in self._model_paths:
            h = self._ov_infer(norm1_model,
                                {"feats": feats.astype(np.float32)})[0]
        else:
            n1_w = W.get(f"{prefix}.norm1.weight")
            n1_b = W.get(f"{prefix}.norm1.bias")
            h = _silu(_layer_norm(feats, n1_w, n1_b))

        # ── conv1 via OV extension ──
        conv1_key = f"{prefix}.conv1"
        h = self._run_subm_conv3d(conv1_key, h, coords)

        # ── norm2 (no affine) + scale/shift + SiLU via inline numpy ──
        norm2_model = f"slat_io_{safe_name}_norm2_act"
        if norm2_model in self._model_paths:
            h = self._ov_infer(norm2_model, {
                "feats":  h.astype(np.float32),
                "scale":  scale.astype(np.float32),
                "shift":  shift.astype(np.float32),
            })[0]
        else:
            # Inline: LayerNorm(h) * (1 + scale) + shift → SiLU
            h_f = h.astype(np.float32)
            mean = h_f.mean(axis=-1, keepdims=True)
            std = np.sqrt(h_f.var(axis=-1, keepdims=True) + 1e-5)
            h_f = (h_f - mean) / std
            h_f = h_f * (1.0 + scale) + shift
            h = (h_f / (1.0 + np.exp(-h_f))).astype(np.float32)  # SiLU

        # ── conv2 via OV extension ──
        conv2_key = f"{prefix}.conv2"
        h = self._run_subm_conv3d(conv2_key, h, coords)

        # ── Skip connection via OV ──
        skip_model = f"slat_io_{safe_name}_skip"
        if skip_model in self._model_paths:
            skip = self._ov_infer(skip_model,
                                   {"feats": feats.astype(np.float32)})[0]
        elif f"{prefix}.skip_connection.weight" in W:
            sk_w = W[f"{prefix}.skip_connection.weight"]
            sk_b = W[f"{prefix}.skip_connection.bias"]
            skip = _linear(feats, sk_w, sk_b)
        else:
            skip = feats  # identity (channels already match)

        return skip + h

    def _gather_concat_parallel(self, x_attn, ds_idx, x2_up_buf, dst, C_attn):
        """Parallel, numerically-identical equivalent of:
             np.take(x_attn, ds_idx, axis=0, out=dst[:, :C_attn])
             dst[:, C_attn:] = x2_up_buf

        Splits the N_vox rows across a persistent thread pool. Both the gather
        (np.take) and the strided copy release the GIL, and each thread writes a
        disjoint contiguous row range, so this is a pure wall-time win on the
        many-core host with no change in results. Falls back to serial for small
        N or when threading is disabled (SAM3D_UPDOWN_WORKERS=1).

        When ``dst`` is float16 (the GEMM path's out_blocks.0 input), the f32->f16
        cast is fused into the gather/copy via assignment (np.take's out= requires
        a matching dtype, so plain fancy-index assignment is used instead). This
        removes the separate 273MB host cast that _to_f16_parallel would do.
        """
        N = dst.shape[0]
        f16 = dst.dtype == np.float16
        nw = getattr(self, "_updown_workers", 1)
        if nw <= 1 or N < 4096:
            if f16:
                # Assignment auto-casts the gathered/copied f32 source to f16.
                dst[:, :C_attn] = x_attn[ds_idx]
            else:
                np.take(x_attn, ds_idx, axis=0, out=dst[:, :C_attn])
            dst[:, C_attn:] = x2_up_buf
            return
        ex = self._updown_pool
        chunk = (N + nw - 1) // nw

        def _work(r0, r1):
            if f16:
                dst[r0:r1, :C_attn] = x_attn[ds_idx[r0:r1]]
            else:
                np.take(x_attn, ds_idx[r0:r1], axis=0, out=dst[r0:r1, :C_attn])
            dst[r0:r1, C_attn:] = x2_up_buf[r0:r1]

        futs = [ex.submit(_work, i, min(i + chunk, N)) for i in range(0, N, chunk)]
        for f in futs:
            f.result()

    def _sample_slat(self, cond: np.ndarray, coords: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
        """Sample structured latent using FlowMatching 12-step ODE.

        IO blocks run in pure Python/NumPy (submanifold sparse conv).
        Attention blocks run via OpenVINO.

        Architecture per SLAT step:
          1. input_layer:   Linear(8   → 128)           Python
          2. input_blocks.0: SparseResBlock3d(128→128)  Python
          3. input_blocks.1: SparseResBlock3d(128→1024) Python  [+ skip_connection]
          4. slat_attn_blocks:  OV model  (N_vox, 1024) → (N_vox, 1024)
          5. out_blocks.0: SparseResBlock3d(2048→128)   Python  (cat attn + skip3)
          6. out_blocks.1: SparseResBlock3d(256→128)    Python  (cat out0 + skip2)
          7. out_layer:    Linear(128 → 8)              Python

        Args:
            cond:   (1, N_cond, 1024)
            coords: (N_vox, 4) int32 — [batch_idx, z, y, x]

        Returns:
            features: (N_vox, 8)
        """
        cfg = self.cfg
        N_vox     = coords.shape[0]
        lat_ch    = cfg.get("slat_latent_channels", 8)
        steps     = cfg.get("slat_inference_steps", 12)
        rescale_t = cfg.get("slat_rescale_t", 1.0)
        time_scale = cfg.get("slat_time_scale", 1000.0)
        grid_size  = cfg.get("slat_grid_size", 64)
        cfg_strength = cfg.get("slat_cfg_strength", 1.0)
        cfg_interval = cfg.get("slat_cfg_interval", [0, 500])

        # coords for sparse IO: [batch,z,y,x] → take z,y,x only (single batch)
        coords_3d = coords[:, 1:].astype(np.int32)   # (N_vox, 3)

        # ── Downsample topology (baseline SparseDownsample(2) before attention) ──
        # The SLAT flow backbone pools voxels 2x (64^3 → 32^3) at input_blocks[1]
        # and restores them at out_blocks[0].  coords are constant across ODE
        # steps, so the grouping (coords//2, unique) is prepared once.
        # ds_idx[n] = group id of voxel n; ds_counts[m] = members in group m;
        # ds_coords_3d[m] = the //2 coordinate of group m (for attention pos-emb).
        ds_c = coords_3d // 2
        ds_key = (ds_c[:, 0].astype(np.int64) * (grid_size * grid_size)
                  + ds_c[:, 1].astype(np.int64) * grid_size
                  + ds_c[:, 2].astype(np.int64))
        _uniq, ds_idx, ds_counts = np.unique(ds_key, return_inverse=True,
                                             return_counts=True)
        ds_idx = ds_idx.astype(np.int64).reshape(-1)          # (N_vox,)
        N_ds = int(_uniq.shape[0])
        # representative //2 coords per group (first occurrence)
        _first = np.zeros(N_ds, dtype=np.int64)
        _first[ds_idx[::-1]] = np.arange(N_vox - 1, -1, -1)   # first index per group
        ds_coords_3d = ds_c[_first].astype(np.int32)          # (N_ds, 3)
        ds_counts_f = ds_counts.astype(np.float32).reshape(N_ds, 1)
        self._log(f"  SLAT voxel pooling: {N_vox} → {N_ds} (2x downsample) "
                  f"for {steps}-step attention")

        # FlowMatching schedule: t goes 0 → 1 (generation direction).
        ts = flowmatching_schedule(steps, rescale_t)  # [0 … 1]

        cond_f32 = cond.astype(np.float32)

        # CFG uses force_zeros_cond=True, so the unconditional pass sees an
        # all-zeros condition.  Every cross-attention key/value then collapses to
        # the projection bias, making the output bit-identical for any cond
        # length — feed a single token instead of ~5.5k.
        uncond_tok = np.zeros((1, 1, cond_f32.shape[2]), np.float32)

        # Initial noise
        feats = rng.standard_normal((N_vox, lat_ch)).astype(np.float32)

        # ── Precompute all timestep embeddings upfront ──
        # t_embedder + IO emb outputs depend only on t_scaled, which is known for
        # every step before the loop, so they can be lifted out of it.
        _t_emb_cache = {}
        for _si in range(steps):
            _ts_scaled = float(ts[_si]) * time_scale
            _t_emb_cache[_si] = self._compute_slat_t_emb(_ts_scaled)

        # Persistent thread pool for the updown gather/concat (CPU-bound, GIL-free
        # numpy ops on ~136MB arrays). Split across host cores for a wall-time win.
        if not hasattr(self, "_updown_pool"):
            from concurrent.futures import ThreadPoolExecutor
            self._updown_workers = int(os.environ.get("SAM3D_UPDOWN_WORKERS", "8"))
            self._updown_pool = ThreadPoolExecutor(max_workers=max(1, self._updown_workers))

        for step_idx in range(steps):
            t_cur  = float(ts[step_idx])
            t_next = float(ts[step_idx + 1])
            dt     = t_next - t_cur
            self._log(f"    SLAT step {step_idx+1}/{steps}: t={t_cur:.3f}")

            t_scaled = t_cur * time_scale
            t_arr    = np.array([t_scaled], dtype=np.float32)

            # 1. input_layer
            x0 = self._ov_infer("slat_input_layer",
                                 {"feats": feats.astype(np.float32)})[0]
            # 2-3. input_blocks (use pre-computed emb outputs from merged model)
            t_emb, emb_in0, emb_in1, emb_out0, emb_out1 = _t_emb_cache[step_idx]
            x1 = self._slat_sparse_resblock(x0, coords_3d, emb_in0,
                                             "input_blocks.0", grid_size)
            # input_blocks.1 has downsample=True → pool voxels 2x, then convs @32^3
            x1_ds = self._ov_infer("slat_downsample", {
                "feats":  x1.astype(np.float32),
                "idx":    ds_idx,
                "counts": ds_counts_f,
            })[0]
            x2 = self._slat_sparse_resblock(x1_ds, ds_coords_3d, emb_in1,
                                             "input_blocks.1", grid_size)

            # ── Pre-allocate upsample buffers once per image ──
            C_attn = x2.shape[1]  # 1024
            if not hasattr(self, '_slat_up_bufs') or self._slat_up_bufs[0].shape != (N_vox, C_attn):
                self._slat_up_bufs = (
                    np.empty((N_vox, C_attn), np.float32),   # x2_up
                )
            x2_up_buf, = self._slat_up_bufs

            # Pre-allocate persistent OV Tensor buffers for attn inputs (once per image)
            # to avoid creating new 24MB Tensor objects every step.
            if not hasattr(self, '_attn_x2_buf') or self._attn_x2_buf.shape[0] != N_ds:
                self._attn_x2_buf      = np.zeros((N_ds, C_attn), np.float32)
                self._attn_coords_buf  = ds_coords_3d.astype(np.int64).copy()
                # Pre-set static attn inputs once (coords constant across all steps)
                for _rk in ("cond", "uncond"):
                    _r = self._get_attn_request(_rk)
                    _r.set_input_tensor(1, ov.Tensor(self._attn_coords_buf))
                    _r.set_input_tensor(3, ov.Tensor(cond_f32 if _rk == "cond" else uncond_tok))
                # shared_memory=True is REQUIRED: ov.Tensor(ndarray) otherwise
                # snapshots the array at construction time, so later in-place
                # writes to the buffer would never reach the model.
                self._attn_x2_ov_cond   = ov.Tensor(self._attn_x2_buf, shared_memory=True)
                self._attn_x2_ov_uncond = ov.Tensor(self._attn_x2_buf, shared_memory=True)
            attn_x2_buf = self._attn_x2_buf

            # After out0's tight resblock is compiled (first step), grab its feat_buf
            # directly so we can gather into it and skip the 273MB copy.
            def _get_out0_feat_buf():
                fk = "slat_io_out_blocks_0_fused_resblock"
                tk = (fk, N_vox)
                if hasattr(self, '_tight_resblock_cache') and tk in self._tight_resblock_cache:
                    req, feat_buf, _, _ = self._tight_resblock_cache[tk]
                    return feat_buf  # (N_vox, 2048)
                return None

            # 4. attention (conditioned) — runs on the pooled 32^3 voxels.
            # Submit ASYNC so CPU can overlap x2 upsample with GPU attn.
            req_cond = self._get_attn_request("cond")
            # Zero-copy: write x2 into pre-allocated buffer that's already set on req
            attn_x2_buf[:] = x2
            req_cond.set_input_tensor(0, self._attn_x2_ov_cond)
            req_cond.set_input_tensor(2, ov.Tensor(t_arr))
            req_cond.start_async()
            # CPU work while GPU runs attn: scatter x2 to full resolution
            x2_up_buf[:] = x2[ds_idx]   # (N_ds,C) → (N_vox,C) gather, ~7ms
            req_cond.wait()
            x_attn_cond = np.array(req_cond.get_output_tensor(0).data, copy=False)

            # 5-7. out_blocks + out_layer → cond velocity
            # Gather directly into out0's pre-allocated feat_buf to eliminate 273MB copy.
            out0_feat_buf = _get_out0_feat_buf()
            if out0_feat_buf is not None:
                # Extension path: gather attn+x2 directly into out0's f32 OV input.
                self._gather_concat_parallel(x_attn_cond, ds_idx, x2_up_buf,
                                             out0_feat_buf, C_attn)
                x_cat_for_out0 = out0_feat_buf  # no copy needed, already in place
            elif getattr(self, "_use_gemm_conv", None) or \
                    (self._use_gemm_conv is None and os.environ.get("SAM3D_GEMM_CONV", "1") != "0"):
                # GEMM path (default): gather+concat straight into a persistent
                # f16 buffer, fusing the f32->f16 cast into the gather so the
                # 273MB out_blocks.0 input never needs a separate host cast.
                buf = getattr(self, "_out0_cat_f16", None)
                if buf is None or buf.shape != (N_vox, C_attn * 2):
                    buf = np.empty((N_vox, C_attn * 2), np.float16)
                    self._out0_cat_f16 = buf
                self._gather_concat_parallel(x_attn_cond, ds_idx, x2_up_buf,
                                             buf, C_attn)
                x_cat_for_out0 = buf
            else:
                # Fallback (extension path, first step before out0 is compiled)
                x_cat_for_out0 = np.empty((N_vox, C_attn * 2), np.float32)
                np.take(x_attn_cond, ds_idx, axis=0, out=x_cat_for_out0[:, :C_attn])
                x_cat_for_out0[:, C_attn:] = x2_up_buf

            # Pre-start uncond attn ASYNC to overlap with cond resblocks.
            # attn_uncond runs 363ms on OV GPU while ext GPU does out0_cond (438ms).
            # The extension uses a DIFFERENT GPU queue, so they can run in parallel.
            use_cfg_here = (cfg_interval[0] <= t_scaled <= cfg_interval[1]) and (cfg_strength != 0.0)
            if use_cfg_here:
                req_uncond = self._get_attn_request("uncond")
                req_uncond.set_input_tensor(0, self._attn_x2_ov_uncond)
                req_uncond.set_input_tensor(2, ov.Tensor(t_arr))
                req_uncond.start_async()
                self._uncond_attn_started = True
            else:
                self._uncond_attn_started = False

            x3 = self._slat_sparse_resblock(x_cat_for_out0, coords_3d, emb_out0,
                                             "out_blocks.0", grid_size)
            x_cat1 = np.concatenate([x3, x1], axis=-1)
            x4 = self._slat_sparse_resblock(x_cat1, coords_3d, emb_out1,
                                             "out_blocks.1", grid_size)
            # Cache slat_out_layer request to avoid new-request-per-call overhead
            if not hasattr(self, '_out_layer_req'):
                m_ol = self._get_model('slat_out_layer')
                self._out_layer_req = m_ol.create_infer_request()
            self._out_layer_req.set_input_tensor(0, ov.Tensor(x4.astype(np.float32)))
            self._out_layer_req.infer()
            v_cond = np.array(self._out_layer_req.get_output_tensor(0).data)

            # CFG
            use_cfg = (cfg_interval[0] <= t_scaled <= cfg_interval[1]) and (cfg_strength != 0.0)
            if use_cfg:
                # Start uncond attn ASYNC concurrently with cond out-blocks above.
                # (moved here from below so GPU runs attn while we did cond resblocks)
                # If this is the first visit, we couldn't have pre-started — start now.
                if not getattr(self, '_uncond_attn_started', False):
                    req_uncond = self._get_attn_request("uncond")
                    req_uncond.set_input_tensor(0, self._attn_x2_ov_uncond)
                    req_uncond.set_input_tensor(2, ov.Tensor(t_arr))
                    req_uncond.start_async()
                req_uncond.wait()
                self._uncond_attn_started = False
                x_attn_uncond = np.array(req_uncond.get_output_tensor(0).data, copy=False)
                # Gather directly into out0's feat_buf (same as cond path)
                if out0_feat_buf is not None:
                    self._gather_concat_parallel(x_attn_uncond, ds_idx, x2_up_buf,
                                                 out0_feat_buf, C_attn)
                    x_cat_for_out0u = out0_feat_buf
                elif getattr(self, "_out0_cat_f16", None) is not None:
                    # GEMM path: reuse the persistent f16 buffer (cond's out0
                    # resblock has already fully consumed it by now).
                    buf = self._out0_cat_f16
                    self._gather_concat_parallel(x_attn_uncond, ds_idx, x2_up_buf,
                                                 buf, C_attn)
                    x_cat_for_out0u = buf
                else:
                    x_cat_for_out0u = np.empty((N_vox, C_attn * 2), np.float32)
                    np.take(x_attn_uncond, ds_idx, axis=0, out=x_cat_for_out0u[:, :C_attn])
                    x_cat_for_out0u[:, C_attn:] = x2_up_buf
                x3u = self._slat_sparse_resblock(x_cat_for_out0u, coords_3d, emb_out0,
                                                  "out_blocks.0", grid_size)
                x_cat1u = np.concatenate([x3u, x1], axis=-1)
                x4u = self._slat_sparse_resblock(x_cat1u, coords_3d, emb_out1,
                                                  "out_blocks.1", grid_size)
                self._out_layer_req.set_input_tensor(0, ov.Tensor(x4u.astype(np.float32)))
                self._out_layer_req.infer()
                v_uncond = np.array(self._out_layer_req.get_output_tensor(0).data)
                velocity = (1.0 + cfg_strength) * v_cond - cfg_strength * v_uncond
                self._log(f"      CFG applied (t_scaled={t_scaled:.1f})")
            else:
                velocity = v_cond
                self._log(f"      CFG skipped (t_scaled={t_scaled:.1f})")

            # Euler update
            feats = feats + dt * velocity

        return feats

    # ─── Main run method ─────────────────────────────────────────────────────

    def run(self, image_path: str, seed: int = 42, mask_path: Optional[str] = None) -> Dict[str, Any]:
        """Run the full SAM3D pipeline on a single image.

        Returns dict with:
          - coords: active voxel coordinates (N, 3)
          - ss_latent: SS latent dict
          - ss_occupancy: occupancy volume (1, 1, 64, 64, 64)
          - slat_features: SLAT features (N_vox, 8)  [if SLAT ran successfully]
        """
        t0 = time.time()
        rng = np.random.default_rng(seed)

        self._log(f"\n{'='*60}")
        self._log(f"Input: {image_path}")
        self._log(f"{'='*60}")

        # ── 1. Load image ──────────────────────────────────────────────
        self._stage("Step 1: Loading image...")
        rgba = load_image_rgba(image_path, mask_path)  # (H, W, 4)
        rgb_full = rgba[:, :, :3]           # (H, W, 3)
        mask_full = rgba[:, :, 3]           # (H, W), alpha as mask

        # ── 2. Compute pointmap via MoGe ──────────────────────────────
        self._stage("Step 2: Computing pointmap (MoGe)...")
        pointmap_hw3 = self._get_pointmap(rgb_full)  # (3, H, W) or None
        if pointmap_hw3 is not None:
            self._log(f"  Pointmap shape: {pointmap_hw3.shape}")
        else:
            self._log("  Pointmap unavailable — PointPatchEmbed will use zeros")

        # ── 3. Preprocess for SS conditioning ─────────────────────────
        self._stage("Step 3: Preprocessing images...")
        dino_size = self.cfg.get("dino_input_size", 518)

        # Crop around mask for "cropped" variants
        y1_c, x1_c, crop_size = get_crop_bbox(mask_full)
        rgb_crop     = apply_crop(rgb_full,  y1_c, x1_c, crop_size)
        mask_crop_hw = apply_crop(mask_full, y1_c, x1_c, crop_size)
        rgb_crop_sq, mask_crop_sq = pad_to_square_centered(rgb_crop, mask_crop_hw)

        # Full variants: baseline img_transform/mask_transform = pad_to_square_centered
        # then Resize(518).  The full image is NOT square, so it must be padded to a
        # centered square BEFORE resizing, else the aspect ratio is distorted.
        rgb_full_sq, mask_full_sq = pad_to_square_centered(rgb_full, mask_full)

        # Preprocess
        img_crop_nchw  = preprocess_for_dino(rgb_crop_sq, dino_size)    # (1,3,518,518)
        img_full_nchw  = preprocess_for_dino(rgb_full_sq, dino_size)    # (1,3,518,518)
        mask_crop_out  = preprocess_mask_for_dino(mask_crop_sq, dino_size)   # (1,1,518,518)
        mask_full_out  = preprocess_mask_for_dino(mask_full_sq, dino_size)   # (1,1,518,518)

        # For pointmap: SSI-normalize at full res (single scale/shift from full mask),
        # then produce crop + full variants matching baseline preprocessor.
        if pointmap_hw3 is not None:
            pm_norm, pm_scale, pm_shift = ssi_normalize_pointmap(pointmap_hw3, mask_full)
            self._log(f"  Pointmap SSI scale={pm_scale[0]:.4f} shift={pm_shift.round(4).tolist()}")

            # Crop variant: crop with same bbox as RGB (NaN out-of-bounds), resize to dino_size
            pm_crop_sq = apply_crop(pm_norm, y1_c, x1_c, crop_size,
                                    pad_val=float('nan'), is_chw=True)
            pointmap_crop_518 = self._resize_nearest_3d(pm_crop_sq, dino_size)
            # Full variant: pad to centered square, resize to dino_size.
            # The pointmap must be padded with 0, NOT NaN: pointmap_transform is
            # Compose([pad_to_square_centered, Resize(518, nearest)]) and the
            # pointmap is passed as the `image` argument, so it takes value=0.
            pm_full_sq = pad_to_square_chw(pm_norm, pad_val=0.0)
            pointmap_full_518 = self._resize_nearest_3d(pm_full_sq, dino_size)
        else:
            pointmap_crop_518 = None
            pointmap_full_518 = None


        # ── 4. Embed conditions (SS) ───────────────────────────────────
        self._stage("Step 4: Running SS EmbedderFuser...")
        ss_cond = self._embed_conditions_ss(
            image_crop=img_crop_nchw[0],     # (3, S, S)
            mask_crop=mask_crop_out[0],       # (1, S, S)
            image_full=img_full_nchw[0],
            mask_full=mask_full_out[0],
            pointmap_crop=pointmap_crop_518,
            pointmap_full=pointmap_full_518,
        )
        self._log(f"  SS condition tokens shape: {ss_cond.shape}")  # (1, 7544, 1024)
        if os.environ.get("DUMP_SS_COND"):
            np.save(os.environ["DUMP_SS_COND"], ss_cond)
            self._log(f"  [DUMP] SS cond -> {os.environ['DUMP_SS_COND']}")

        # ── 5+7. SS ODE + SLAT embed (overlapped) ────────────────────
        # SLAT embed only needs the preprocessed images (ready now), not voxel
        # coords. Launch it on a background thread so it runs during SS ODE
        # (15-16s), hiding its ~0.9s cost entirely.
        from concurrent.futures import ThreadPoolExecutor as _TPE
        _slat_embed_ex = _TPE(max_workers=1)
        _slat_embed_fut = _slat_embed_ex.submit(
            self._embed_conditions_slat,
            img_full_nchw[0], mask_full_out[0],
            img_crop_nchw[0], mask_crop_out[0],
        )

        self._stage(f"Step 5: Running SS ShortCut ODE ({self.cfg.get('ss_inference_steps', 25)} steps)...")
        ss_latent_dict = self._sample_ss(ss_cond, rng)
        shape_latent = ss_latent_dict["shape"]  # (1, 4096, 8)
        self._log(f"  SS shape latent: {shape_latent.shape}")

        # ── 6. Decode SS → occupancy → coordinates ─────────────────────
        self._stage("Step 6: Decoding SS → occupancy...")
        occupancy = self._decode_ss(shape_latent)  # (1, 1, 64, 64, 64)
        occupied = (occupancy[0, 0] > 0).astype(np.int32)
        coords_zyx = np.argwhere(occupied)         # (N_vox, 3): z, y, x
        N_vox = coords_zyx.shape[0]
        self._log(f"  Active voxels: {N_vox}")

        if N_vox == 0:
            self._log("  WARNING: No active voxels after SS decoding!")
            _slat_embed_ex.shutdown(wait=False)
            return {
                "coords": coords_zyx,
                "ss_latent": ss_latent_dict,
                "ss_occupancy": occupancy,
            }

        # Add batch index column: (N_vox, 4) = [batch_idx, z, y, x]
        coords_batch = np.concatenate([
            np.zeros((N_vox, 1), dtype=np.int32),
            coords_zyx,
        ], axis=1)

        # Baseline parity: downsample sparse structure when > max_coords voxels
        # (inference_pipeline.sample_ss). No-op below the threshold.
        coords_batch, ss_downsample_factor = downsample_sparse_structure(coords_batch)
        if ss_downsample_factor != 1:
            N_vox = coords_batch.shape[0]
            coords_zyx = coords_batch[:, 1:]
            self._log(f"  Downsampled coords to {N_vox} (factor {ss_downsample_factor})")

        # ── 7. Collect SLAT embed result (was running in background) ───
        self._stage("Step 7: Running SLAT EmbedderFuser...")
        slat_cond = _slat_embed_fut.result()   # already done if SS ODE took >0.9s
        _slat_embed_ex.shutdown(wait=False)
        self._log(f"  SLAT condition tokens shape: {slat_cond.shape}")

        # ── 8. Sample SLAT ────────────────────────────────────────────
        self._stage(f"Step 8: Running SLAT FlowMatching ODE ({self.cfg.get('slat_inference_steps', 25)} steps)...")
        import time as _time_mod
        _slat_t0 = _time_mod.perf_counter()
        try:
            slat_feats = self._sample_slat(slat_cond, coords_batch, rng)  # (N_vox, 8)
            _slat_t1 = _time_mod.perf_counter()
            self._log(f"  SLAT ODE wall time: {_slat_t1 - _slat_t0:.2f}s")
            # Unnormalize
            slat_mean = np.array(self.cfg["slat_mean"], dtype=np.float32)
            slat_std  = np.array(self.cfg["slat_std"], dtype=np.float32)
            slat_feats_unnorm = slat_feats * slat_std[None] + slat_mean[None]
            self._log(f"  SLAT features: {slat_feats_unnorm.shape}")
        except NotImplementedError as e:
            self._log(f"  SLAT sampling incomplete: {e}")
            slat_feats_unnorm = None

        # ── 9. Decode SLAT → Gaussians + mesh feats (in parallel) ───
        # Both decoders need only slat_feats_unnorm + coords_zyx (read-only), so
        # they run on separate threads with distinct OV model instances.
        gaussians = None
        mesh_hidden_feats = None
        if slat_feats_unnorm is not None:
            self._stage("Step 9: Decoding SLAT → Gaussian parameters + mesh features (parallel)...")
            from concurrent.futures import ThreadPoolExecutor as _TPE2
            with _TPE2(max_workers=2) as _dec_ex:
                _gs_fut   = _dec_ex.submit(self._decode_slat_gs,
                                           slat_feats_unnorm, coords_zyx)
                _mesh_fut = _dec_ex.submit(self._decode_slat_mesh_feats,
                                           slat_feats_unnorm, coords_zyx)
                gaussians         = _gs_fut.result()
                mesh_hidden_feats = _mesh_fut.result()
            if gaussians is not None:
                self._log(f"  Gaussians decoded: {gaussians['_xyz'].shape[0]} total")
            else:
                self._log("  Gaussian decoder not available")
            if mesh_hidden_feats is not None:
                self._log(f"  Mesh hidden feats: {mesh_hidden_feats.shape}")
            else:
                self._log("  Mesh decoder not available")

        # ── 10. Extract mesh (sparse subdivide upsample + FlexiCubes) ──
        # The upsample (GPU op) runs now; FlexiCubes (CPU op) is launched on a
        # background thread so it overlaps with the Gaussian PLY export in main().
        # Record end-to-end time up to (and including) the Gaussian decode, i.e.
        # the point at which a Gaussian-splat PLY can be saved.
        t_no_mesh = time.time() - t0

        mesh = None
        mesh_future = None
        if mesh_hidden_feats is not None:
            self._stage("Step 10: Extracting mesh (upsample + FlexiCubes)...")
            cube = self._mesh_upsample(mesh_hidden_feats, coords_zyx)
            if cube is not None:
                from concurrent.futures import ThreadPoolExecutor
                self._mesh_executor = ThreadPoolExecutor(max_workers=1)
                mesh_future = self._mesh_executor.submit(self._flexicubes_extract, *cube)

        return {
            "coords": coords_zyx,               # (N_vox, 3)
            "ss_latent": ss_latent_dict,
            "ss_occupancy": occupancy,
            "slat_features": slat_feats_unnorm, # (N_vox, 8)
            "gaussians": gaussians,             # dict of Gaussian param arrays, or None
            "mesh_hidden_feats": mesh_hidden_feats,  # (N_vox, 768), or None
            "mesh": mesh,                       # resolved mesh dict (None until future done)
            "mesh_future": mesh_future,         # Future[dict] running FlexiCubes, or None
            "t_no_mesh": t_no_mesh,             # wall time to SLAT PLY (excl. mesh)
        }

    @staticmethod
    def _resize_nearest_3d(arr: np.ndarray, size: int) -> np.ndarray:
        """Resize (C, H, W) to (C, size, size) by nearest neighbor."""
        C, H, W = arr.shape
        ys = (np.arange(size) * H / size).astype(int)
        xs = (np.arange(size) * W / size).astype(int)
        return arr[:, ys[:, None], xs[None, :]]

    # ─── SLAT Gaussian Decoder ────────────────────────────────────────────────

    @staticmethod
    def _hammersley_perturbation(num_gaussians: int, voxel_size: float) -> np.ndarray:
        """Compute Hammersley perturbation offsets, matching baseline _build_perturbation.

        Returns (num_gaussians, 3) array of atanh offsets.
        """
        PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        def radical_inverse(base: int, n: int) -> float:
            val, inv_base_n = 0.0, 1.0 / base
            while n > 0:
                digit = n % base
                val += digit * inv_base_n
                n //= base
                inv_base_n /= base
            return val

        def hammersley(dim: int, n: int, num_samples: int):
            return [n / num_samples] + [radical_inverse(PRIMES[d], n) for d in range(dim - 1)]

        pts = np.array([hammersley(3, i, num_gaussians) for i in range(num_gaussians)],
                       dtype=np.float32)   # (G, 3) in [0,1]
        pts = pts * 2 - 1                  # → [-1, 1]
        pts = pts / voxel_size             # scale
        pts = np.arctanh(np.clip(pts, -1 + 1e-6, 1 - 1e-6))  # atanh
        return pts  # (G, 3)

    def _windowed_decoder_gs_forward(self, slat_feats: np.ndarray,
                                      coords_zyx: np.ndarray) -> np.ndarray:
        """Run GS decoder as a SINGLE Swin-windowed OV model (slat_dec_gs).

        Swin windowing (window_size=8, alternating shift 0/4 on even/odd blocks)
        is done inside the OV graph. Here we only build the deterministic gather /
        inverse-gather / mask index tensors from voxel coordinates and pass them
        in. Numerically identical to per-window full attention (pad keys masked).

        Layout for one parity:
          - assign each voxel to its window id, group voxels by window
          - pad every window to max occupancy (mo); pad slots reference row N
          - perm: (nw*mo,) gather indices into [0..N]  (N = zero pad row)
          - inv:  (N,) inverse — for each voxel, its slot in the flattened windows
          - mask: (nw, mo) additive attn mask (0 valid, -3e4 pad)
        """
        N = slat_feats.shape[0]
        window_size = 8
        grid_size = 64
        gw = grid_size // window_size  # 8
        coords_int = coords_zyx.astype(np.int64)

        def _build(shift: int):
            shifted = coords_int + shift
            wids = shifted // window_size
            linear = wids[:, 0] * gw * gw + wids[:, 1] * gw + wids[:, 2]
            order = np.argsort(linear, kind="stable")
            lin_sorted = linear[order]
            uniq, starts, counts = np.unique(lin_sorted, return_index=True,
                                             return_counts=True)
            nw = len(uniq)
            mo = int(counts.max()) if nw > 0 else 1
            # perm: (nw, mo) filled with pad index N, then valid voxels
            perm = np.full((nw, mo), N, dtype=np.int64)
            inv = np.zeros(N, dtype=np.int64)
            for w in range(nw):
                s = starts[w]; c = counts[w]
                vox = order[s:s + c]            # original voxel indices in window w
                perm[w, :c] = vox
                inv[vox] = w * mo + np.arange(c)
            mask = np.zeros((nw, mo), dtype=np.float32)
            for w in range(nw):
                mask[w, counts[w]:] = -3e4
            return perm.reshape(-1), inv, mask

        perm_e, inv_e, mask_e = _build(0)
        perm_o, inv_o, mask_o = _build(window_size // 2)

        out = self._ov_infer("slat_dec_gs", {
            "feats":  slat_feats.astype(np.float32),
            "coords": coords_zyx.astype(np.float32),
            "perm_e": perm_e, "inv_e": inv_e, "mask_e": mask_e,
            "perm_o": perm_o, "inv_o": inv_o, "mask_o": mask_o,
        })
        return out[0]  # (N, 448)

    def _decode_slat_gs(self, slat_feats: np.ndarray,
                        coords_zyx: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """Decode unnormalized SLAT features → per-voxel Gaussian parameters.

        Returns dict with Gaussian parameter arrays, or None if model unavailable.

        The baseline decoder uses Swin-style windowed attention (window_size=8)
        with shifted windows on odd blocks. Since the OV-exported decoder uses
        full attention, we approximate windowed attention by processing spatial
        windows independently through the decoder.

        Output layout (matches SLatGaussianDecoder._calc_layout()):
            _xyz:         (N_vox * 32, 3)
            _features_dc: (N_vox * 32, 3)
            _scaling:     (N_vox * 32, 3)
            _rotation:    (N_vox * 32, 4)
            _opacity:     (N_vox * 32, 1)
        """
        has_gs_decoder = "slat_dec_gs" in self._model_paths
        if not has_gs_decoder:
            self._log("  SLAT GS decoder not registered — skipping Gaussian output")
            return None

        N_vox = slat_feats.shape[0]

        # ── Single Swin-windowed OV decoder (slat_dec_gs) ──
        self._log(f"  Running SLAT GS decoder (single windowed OV model) on {N_vox} voxels...")
        dec_feats = self._windowed_decoder_gs_forward(
            slat_feats.astype(np.float32),
            coords_zyx.astype(np.int32),
        )

        # Read representation config
        cfg = self.cfg
        resolution     = cfg.get("slat_dec_resolution", 64)
        num_gaussians  = cfg.get("slat_dec_num_gaussians", 32)
        voxel_size     = cfg.get("slat_dec_voxel_size", 1.5)
        lr             = cfg.get("slat_dec_gs_lr", {
            "_xyz": 1.0, "_features_dc": 1.0, "_opacity": 1.0,
            "_scaling": 1.0, "_rotation": 0.1,
        })
        perturb_offset = cfg.get("slat_dec_gs_perturb_offset", True)

        # Canonical voxel centres: [0, 1)^3
        # Baseline decoder_gs.py: xyz = (coords[:,1:] + 0.5) / resolution, where
        # coords[:,1:] is [z, y, x].  No axis reversal: voxel-Z is world-X.
        voxel_xyz = (coords_zyx.astype(np.float32) + 0.5) / resolution  # (N,3) [z→x, y→y, x→z]

        # Parse layout: _xyz, _features_dc, _scaling, _rotation, _opacity
        layout = [
            ("_xyz",         num_gaussians * 3),
            ("_features_dc", num_gaussians * 3),
            ("_scaling",     num_gaussians * 3),
            ("_rotation",    num_gaussians * 4),
            ("_opacity",     num_gaussians),
        ]
        shapes = {
            "_xyz":         (num_gaussians, 3),
            "_features_dc": (num_gaussians, 1, 3),
            "_scaling":     (num_gaussians, 3),
            "_rotation":    (num_gaussians, 4),
            "_opacity":     (num_gaussians, 1),
        }
        result: Dict[str, np.ndarray] = {}
        start = 0
        for name, size in layout:
            chunk = dec_feats[:, start:start + size].reshape(N_vox, *shapes[name])
            chunk = chunk * lr[name]
            if name == "_xyz":
                if perturb_offset:
                    # Hammersley perturbation: same as baseline _build_perturbation
                    perturb = self._hammersley_perturbation(num_gaussians, voxel_size)  # (G,3)
                    chunk = chunk + perturb[None, :, :]   # broadcast over N_vox
                offset = np.tanh(chunk) / resolution * 0.5 * voxel_size   # (N, 32, 3)
                # Apply aabb: get_xyz = _xyz * aabb[3:] + aabb[:3]
                # aabb = [-0.5, -0.5, -0.5, 1.0, 1.0, 1.0]
                xyz = voxel_xyz[:, None, :] + offset                       # (N, 32, 3) in [0,1]
                xyz = xyz * 1.0 + (-0.5)   # → [-0.5, 0.5] to match baseline aabb
                result["_xyz"] = xyz.reshape(N_vox * num_gaussians, 3)
            else:
                result[name] = chunk.reshape(N_vox * num_gaussians, *shapes[name][1:])
            start += size

        self._log(f"    Gaussians generated: {N_vox * num_gaussians}")
        return result

    def _decode_slat_mesh_feats(self, slat_feats: np.ndarray,
                                 coords_zyx: np.ndarray) -> Optional[np.ndarray]:
        """Run SLAT mesh decoder attn blocks → hidden features (N_vox, 768).

        Note: The SparseSubdivideBlock3d upsample layers (which use sparse conv3d
        and increase spatial resolution) and the final Linear(96→101) are handled
        by the OV extension pipeline via save_slat_decoder_mesh_upsample_weights().
        This method returns the per-voxel hidden features ready for upsampling.
        """
        if "slat_decoder_mesh_attn" not in self._model_paths:
            self._log("  SLAT mesh attn decoder not registered — skipping mesh output")
            return None

        N_vox = slat_feats.shape[0]
        coords_int = coords_zyx.astype(np.int64)

        self._log(f"  Running SLAT mesh attn on {N_vox} voxels...")
        ov_out = self._ov_infer("slat_decoder_mesh_attn", {
            "feats":  slat_feats.astype(np.float32),
            "coords": coords_int,
        })
        return ov_out[0]  # (N_vox, 768)

    # ─── Mesh extraction (sparse subdivide upsample + FlexiCubes) ─────────────
    def _mesh_upsample(self, mesh_hidden_feats: np.ndarray,
                       coords_zyx: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Stage 2: sparse subdivide upsample (res 64 → 256) via the OV op.

        Returns (cube_features (M, 101), cube_coords (M, 4)) or None if the op
        is not registered.
        """
        if ("slat_mesh_upsample" not in self._model_paths
                or "slat_flexicubes" not in self._model_paths):
            self._log("  Mesh upsample / FlexiCubes ops not registered — skipping mesh extraction")
            return None

        N_vox = mesh_hidden_feats.shape[0]
        coords4 = np.zeros((N_vox, 4), dtype=np.int32)
        coords4[:, 1:] = coords_zyx.astype(np.int32)   # [b=0, z, y, x]

        self._log(f"  [mesh] Running mesh upsample on {N_vox} voxels...")
        _t0 = time.perf_counter()
        up = self._ov_infer("slat_mesh_upsample", {
            "features":   mesh_hidden_feats.astype(np.float32),
            "coords":     coords4,
            "num_voxels": np.array([N_vox], dtype=np.int32),
        })
        cube_feats  = np.asarray(up["out_features"])   # (M, 101)
        cube_coords = np.asarray(up["out_coords"])     # (M, 4)
        out_num     = int(np.asarray(up["out_num"]).reshape(-1)[0])
        cube_feats  = cube_feats[:out_num]
        cube_coords = cube_coords[:out_num]
        self._log(f"  [mesh] Upsampled to {out_num} cube voxels (res 256) "
                  f"in {time.perf_counter() - _t0:.2f}s")
        return cube_feats, cube_coords

    def _flexicubes_extract(self, cube_feats: np.ndarray,
                            cube_coords: np.ndarray) -> Dict[str, np.ndarray]:
        """Stage 8: FlexiCubes / SparseFeatures2Mesh via the OV op (CPU).

        Returns a dict {vertices, faces, colors}.
        """
        out_num = cube_feats.shape[0]
        self._log(f"  [mesh] Running FlexiCubes extraction on {out_num} cube voxels...")
        _t0 = time.perf_counter()
        fc = self._ov_infer("slat_flexicubes", {
            "features":   cube_feats.astype(np.float32),
            "coords":     cube_coords.astype(np.int32),
            "num_voxels": np.array([out_num], dtype=np.int32),
        })
        counts   = np.asarray(fc["counts"]).reshape(-1)
        n_verts, n_faces = int(counts[0]), int(counts[1])
        vertices = np.asarray(fc["vertices"])[:n_verts]   # (V, 3)
        faces    = np.asarray(fc["faces"])[:n_faces]      # (F, 3)
        colors   = np.asarray(fc["colors"])[:n_verts]     # (V, 6)
        self._log(f"  [mesh] Extracted mesh: {n_verts} vertices, {n_faces} faces "
                  f"in {time.perf_counter() - _t0:.2f}s")
        return {"vertices": vertices, "faces": faces, "colors": colors}

    @staticmethod
    def _mesh_to_glb_vertex_color(mesh: Dict[str, np.ndarray], output_path: str) -> None:
        """Assemble a textured GLB using per-vertex colors.

        Mirrors postprocessing_utils.to_glb() with
        with_mesh_postprocess=False, with_texture_baking=False, use_vertex_color=True
        — which is exactly the golden benchmark's GLB path
        (docker-baseline/benchmark.py runs the pipeline with these flags):
          - vert_colors = mesh.vertex_attrs[:, :3]  (float RGB in [0, 1])
          - vertices rotated z-up -> y-up: v @ [[1,0,0],[0,0,-1],[0,1,0]]
          - trimesh converts the float colors to uint8 RGBA internally.
        No mesh post-processing or texture baking (not used by the golden).
        """
        import trimesh
        vertices = mesh["vertices"].astype(np.float32)
        faces    = mesh["faces"].astype(np.int64)
        vert_colors = mesh["colors"][:, :3].astype(np.float32)   # RGB in [0, 1]

        # z-up -> y-up (matches to_glb)
        rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        vertices = vertices @ rot

        tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        tm.visual.vertex_colors = vert_colors
        tm.export(output_path)


# ─── PLY export ──────────────────────────────────────────────────────────────

def save_gaussian_splat_ply(gaussians: Dict[str, np.ndarray], output_path: str,
                             scaling_bias: float = 0.004,
                             opacity_bias: float = 0.1,
                             min_kernel_size: float = 0.0009,
                             scaling_activation: str = "softplus") -> None:
    """Save Gaussian splat to PLY in the format used by the baseline.

    Applies activations to match baseline Gaussian.save_ply():
      - opacity:  inverse_sigmoid(sigmoid(raw + opacity_bias))
      - scaling:  log(sqrt(softplus(raw + scale_bias)^2 + min_kernel^2))
      - rotation: normalized (raw + [1,0,0,0])
      - xyz:      as-is (already in world space via aabb)
      - f_dc:     raw features flattened
    """
    from plyfile import PlyData, PlyElement

    xyz = gaussians["_xyz"].astype(np.float32)         # (M, 3)
    f_dc = gaussians["_features_dc"].astype(np.float32)  # (M, 1, 3) or (M, 3)
    scaling = gaussians["_scaling"].astype(np.float32)  # (M, 3)
    rotation = gaussians["_rotation"].astype(np.float32)  # (M, 4)
    opacity = gaussians["_opacity"].astype(np.float32)   # (M, 1)

    M = xyz.shape[0]

    # Apply biases (matching baseline setup_functions)
    # scale_bias = inverse_scaling_activation(scaling_bias)
    if scaling_activation == "softplus":
        # softplus_inverse: log(exp(x)-1)
        scale_bias = np.log(np.expm1(scaling_bias)).astype(np.float32)
        # get_scaling = sqrt(softplus(raw + scale_bias)^2 + min_kernel^2)
        raw_s = scaling + scale_bias
        act_s = np.log1p(np.exp(raw_s))  # softplus
        active_s = np.sqrt(act_s ** 2 + min_kernel_size ** 2)
        # save as log(get_scaling)
        s_out = np.log(np.clip(active_s, 1e-12, None))
    else:
        scale_bias = np.log(scaling_bias).astype(np.float32)
        raw_s = scaling + scale_bias
        active_s = np.sqrt(np.exp(raw_s) ** 2 + min_kernel_size ** 2)
        s_out = np.log(np.clip(active_s, 1e-12, None))

    # opacity bias
    def inv_sigmoid(x):
        return np.log(x / (1 - x))
    op_bias = inv_sigmoid(np.clip(opacity_bias, 1e-6, 1 - 1e-6))
    act_op = 1.0 / (1.0 + np.exp(-(opacity + op_bias)))  # sigmoid
    op_out = inv_sigmoid(np.clip(act_op, 1e-6, 1 - 1e-6))

    # rotation: add rots_bias [1,0,0,0] and normalize
    rots_bias = np.array([1., 0., 0., 0.], dtype=np.float32)
    rot_raw = rotation + rots_bias[None, :]
    norm = np.linalg.norm(rot_raw, axis=1, keepdims=True)
    rot_out = rot_raw / np.clip(norm, 1e-12, None)

    # f_dc: flatten (M,1,3) → (M,3)
    if f_dc.ndim == 3:
        f_dc_out = f_dc.transpose(0, 2, 1).reshape(M, -1)  # (M, 3)
    else:
        f_dc_out = f_dc.reshape(M, -1)

    normals = np.zeros((M, 3), dtype=np.float32)

    dtype_full = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]

    elements = np.empty(M, dtype=dtype_full)
    attrs = np.concatenate([xyz, normals, f_dc_out,
                             op_out.reshape(M, 1), s_out, rot_out], axis=1)  # (M, 17)
    elements[:] = list(map(tuple, attrs))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(output_path)
    print(f"Saved Gaussian splat PLY: {output_path} ({M:,} Gaussians)")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SAM 3D Objects — OpenVINO Standalone Inference"
    )
    parser.add_argument(
        "--model-dir", required=True,
        help="Directory containing exported OV IR models and config.json"
    )
    parser.add_argument(
        "--ext-dir", default=None,
        help="Directory containing OV extension .so libraries. "
             "Default: <model-dir>/../openvino_extensions"
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to input image (RGBA PNG or RGB JPEG)"
    )
    parser.add_argument(
        "--mask", default=None,
        help="Path to mask image (alpha = object region).  If not given, the"
             " script looks for 0.png in the same directory as --image."
    )
    parser.add_argument(
        "--output", default="output.ply",
        help="Output PLY file path"
    )
    parser.add_argument(
        "--device", default="GPU",
        help="OpenVINO inference device (GPU, CPU, AUTO)"
    )
    parser.add_argument(
        "--moge-dir", default=None,
        help="Path to MoGe checkpoint directory (optional)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--slat-steps", type=int, default=None,
        help="Number of SLAT FlowMatching ODE steps (default: value in "
             "config.json, 25). Lower values trade accuracy for latency and "
             "will diverge from the reference baseline."
    )
    args = parser.parse_args()

    if args.ext_dir:
        ext_dir = args.ext_dir
    else:
        # Prefer <model-dir>/../openvino_extensions (works when model_dir is
        # inside the repo).  Fall back to the directory containing this script
        # (works when --model-dir points outside the repo).
        _sibling = Path(args.model_dir).parent / "openvino_extensions"
        _script  = Path(__file__).parent / "openvino_extensions"
        if _sibling.is_dir():
            ext_dir = str(_sibling)
        elif _script.is_dir():
            ext_dir = str(_script)
        else:
            ext_dir = str(_sibling)  # let it fail with a clear path in the log

    # Auto-discover mask: if --mask not given, look for 0.png next to the image.
    # This matches the NVIDIA baseline default (--mask-index 0) and eval_accuracy.py.
    mask_path = args.mask
    if mask_path is None:
        candidate = Path(args.image).parent / "0.png"
        if candidate.exists():
            mask_path = str(candidate)
            print(f"[info] --mask not specified; auto-discovered mask: {mask_path}")
        else:
            print("[warn] No --mask given and no 0.png found next to the image.")
            print("       The entire image will be used as the object region — voxel")
            print("       count will be significantly lower than the NVIDIA baseline.")

    print("=" * 70)
    print("SAM 3D Objects — OpenVINO Standalone Inference")
    print("=" * 70)
    print(f"  Model dir:  {args.model_dir}")
    print(f"  Extension:  {ext_dir}")
    print(f"  Image:      {args.image}")
    print(f"  Mask:       {mask_path or '(none — full image)'}")
    print(f"  Device:     {args.device}")
    print(f"  Seed:       {args.seed}")
    if args.slat_steps is not None:
        print(f"  SLAT steps: {args.slat_steps} (CLI override)")
    print()

    pipeline = SAM3DInference(
        model_dir=args.model_dir,
        ext_dir=ext_dir,
        device=args.device,
        moge_checkpoint_dir=args.moge_dir,
        verbose=not args.quiet,
    )

    if args.slat_steps is not None:
        pipeline.cfg["slat_inference_steps"] = args.slat_steps

    result = pipeline.run(
        image_path=args.image,
        seed=args.seed,
        mask_path=mask_path,
    )

    # Save output
    if result.get("coords") is not None and len(result["coords"]) > 0:
        out_base = args.output
        # 1. Save Gaussian splat PLY — only when the output format is PLY.
        #    When the user requests a .glb mesh, skip this: the 1M-Gaussian PLY
        #    (~2-3s) is an unrequested side-file in that case.
        want_ply = out_base.endswith(".ply")
        if result.get("gaussians") is not None and want_ply:
            cfg = pipeline.cfg
            _t_ply = time.perf_counter()
            save_gaussian_splat_ply(
                result["gaussians"],
                out_base,
                scaling_bias=cfg.get("slat_dec_gs_scaling_bias", 0.004),
                opacity_bias=cfg.get("slat_dec_gs_opacity_bias", 0.1),
                min_kernel_size=0.0009,
                scaling_activation="softplus",
            )
            if not args.quiet:
                print(f"  Saved Gaussian PLY in {time.perf_counter() - _t_ply:.2f}s")
        # 2. Save textured mesh GLB (vertex-color path).  Join the background
        #    FlexiCubes future.
        mesh = result.get("mesh")
        mesh_future = result.get("mesh_future")
        if mesh is None and mesh_future is not None:
            _t_join = time.perf_counter()
            mesh = mesh_future.result()
            if not args.quiet:
                print(f"  Waited {time.perf_counter() - _t_join:.2f}s for FlexiCubes "
                      f"to finish")
        if mesh is not None:
            glb_path = out_base[:-4] + ".glb" if out_base.endswith(".ply") else (out_base if out_base.endswith(".glb") else out_base + ".glb")
            SAM3DInference._mesh_to_glb_vertex_color(mesh, glb_path)
            print(f"  Saved mesh GLB: {glb_path}")
        # Release the background mesh executor (if any)
        _ex = getattr(pipeline, "_mesh_executor", None)
        if _ex is not None:
            _ex.shutdown(wait=True)

        # ── End-to-end timing summary ──────────────────────────────────
        t_no_mesh = result.get("t_no_mesh")
        print("")
        print("=" * 60)
        print(f"  E2E Pipeline time: {t_no_mesh:.1f}s  ")
        print("=" * 60)
    else:
        print("No output produced — check the logs above.")


if __name__ == "__main__":
    main()
