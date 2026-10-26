# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Centralized constants for SAM 3D Objects OpenVINO enablement.

All model architecture constants, diffusion parameters, preprocessing
settings, and sparse conv layer definitions are collected here so that
both export.py and run_inference_standalone.py can share a single source
of truth.

Architecture derived from actual checkpoint configs at:
  checkpoints/hf-download/checkpoints/{pipeline,ss_generator,slat_generator}.yaml
"""

from __future__ import annotations

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# Image Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════
DINO_INPUT_SIZE = 518                        # DINOv2 ViT-L/14-reg input resolution
DINO_PATCH_SIZE = 14                         # ViT-L/14 patch size
DINO_NUM_REGISTERS = 4                       # ViT-L/14-reg register tokens
DINO_NUM_CLS = 1                             # CLS token
DINO_NUM_PATCHES = (DINO_INPUT_SIZE // DINO_PATCH_SIZE) ** 2   # 37² = 1369
DINO_NUM_TOKENS = DINO_NUM_PATCHES + DINO_NUM_CLS + DINO_NUM_REGISTERS  # 1374
DINO_EMBED_DIM  = 1024                       # ViT-L hidden dimension
DINO_MEAN = [0.485, 0.456, 0.406]           # ImageNet channel means
DINO_STD  = [0.229, 0.224, 0.225]           # ImageNet channel stds
PIPELINE_IMAGE_SIZE = 518                    # Full pipeline preprocessing size

# ═══════════════════════════════════════════════════════════════════════════════
# PointPatchEmbed (for SS generator pointmap conditioning)
# ═══════════════════════════════════════════════════════════════════════════════
POINTPATCH_INPUT_SIZE  = 256                 # XYZ map input resolution
POINTPATCH_PATCH_SIZE  = 8                   # Window size
POINTPATCH_EMBED_DIM   = 512                 # Internal embed dim (before projection)
POINTPATCH_NUM_TOKENS  = (POINTPATCH_INPUT_SIZE // POINTPATCH_PATCH_SIZE) ** 2  # 1024
POINTPATCH_NUM_HEADS   = 16                  # Intra-window attention heads

# ═══════════════════════════════════════════════════════════════════════════════
# Condition Embedder — EmbedderFuser (DINOv2 × 2 + PointPatchEmbed for SS)
# ═══════════════════════════════════════════════════════════════════════════════
COND_EMBED_DIM = 1024                        # Output dim of EmbedderFuser projection nets

# SS EmbedderFuser: 2 DINO embedders × 2 kwargs each + 1 PointPatchEmbed × 2 kwargs
#   Embedder-0 (DINO): image (cropped) + rgb_image (full)   → 2 × 1374 tokens
#   Embedder-1 (DINO): mask (cropped) + rgb_image_mask (full) → 2 × 1374 tokens
#   Embedder-2 (PointPatchEmbed): pointmap + rgb_pointmap   → 2 × 1024 tokens
SS_COND_DINO_CALLS  = 4                      # 4 DINO forward passes (image,rgb,mask,rgb_mask)
SS_COND_POINT_CALLS = 2                      # 2 PointPatchEmbed passes (pointmap, rgb_pointmap)
SS_COND_NUM_TOKENS  = SS_COND_DINO_CALLS * DINO_NUM_TOKENS + SS_COND_POINT_CALLS * POINTPATCH_NUM_TOKENS
# = 4 × 1374 + 2 × 1024 = 5496 + 2048 = 7544

# SLAT EmbedderFuser: 2 DINO embedders × 2 kwargs each (no PointPatchEmbed)
SLAT_COND_DINO_CALLS = 4
SLAT_COND_NUM_TOKENS = SLAT_COND_DINO_CALLS * DINO_NUM_TOKENS   # 4 × 1374 = 5496

# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Sparse Structure (SS) — Stage 1
# ═══════════════════════════════════════════════════════════════════════════════
# SS latent: (B, 8, 16, 16, 16) — feeds into Con3D decoder
SS_LATENT_CHANNELS = 8
SS_SPATIAL_RES     = 16                      # 16×16×16 latent grid
SS_VOLUME_RES      = 64                      # Output occupancy grid resolution
SS_NUM_TOKENS      = SS_SPATIAL_RES ** 3     # 4096  (res=16, patch_size=1 → all tokens)

# SS Generator (MM-DiT backbone, 24 blocks, ShortCut ODE)
SS_MODEL_CHANNELS  = 1024
SS_NUM_BLOCKS      = 24
SS_NUM_HEADS       = 16
SS_PATCH_SIZE      = 1                       # No patchification
SS_IS_SHORTCUT     = True                    # ShortCut 2-step ODE (not standard FM)
SS_INFERENCE_STEPS = 2                       # ShortCut uses 2 inference steps
SS_CFG_STRENGTH    = 2.0                     # ClassifierFreeGuidanceWithExternalUnconditional
SS_CFG_INTERVAL    = (0, 500)
SS_RESCALE_T       = 1.0
SS_TIME_SCALE      = 1000.0

# SS latent modalities (MM-DiT with latent_share_transformer)
# shape tokens: (B, 4096, 8) — 16³ flattened shape latent
# pose group tokens after merge: (B, 4, 1024) — [rot(1), trans(1), scale(1), trans_scale(1)]
SS_SHAPE_TOKENS            = SS_NUM_TOKENS   # 4096
SS_POSE_TOKENS_MERGED      = 4               # merged group: rot+trans+scale+trans_scale
SS_SHAPE_CHANNELS          = 8
SS_ROTATION_CHANNELS       = 6               # 6D rotation representation
SS_TRANSLATION_CHANNELS    = 3
SS_SCALE_CHANNELS          = 3
SS_TRANSLATION_SCALE_CHANNELS = 1

# SS Decoder (dense Conv3D VAE)
SS_DECODER_LATENT_CH  = 8                    # Input channels: (B, 8, 16, 16, 16)
SS_DECODER_CHANNELS   = [512, 128, 32]       # From ss_decoder.yaml
SS_DECODER_OUT_CH     = 1                    # Output: occupancy logits (B, 1, 64, 64, 64)

# ═══════════════════════════════════════════════════════════════════════════════
# Structured Latent (SLAT) — Stage 2
# ═══════════════════════════════════════════════════════════════════════════════
SLAT_LATENT_CHANNELS   = 8
SLAT_SPATIAL_RES       = 64                  # 64³ sparse volume resolution
MAX_VOXELS             = 65536               # 64³ upper bound on active voxels

# SLAT Generator (sparse DiT, FlowMatching ODE, no CFG)
SLAT_MODEL_CHANNELS    = 1024
SLAT_NUM_BLOCKS        = 24
SLAT_NUM_HEADS         = 16
SLAT_PATCH_SIZE        = 2                   # Spatial patch factor (stride-2 downsample in IO)
SLAT_IO_BLOCK_CHANNELS = [128]               # io_block_channels from config
SLAT_NUM_IO_RES_BLOCKS = 2                   # num_io_res_blocks
SLAT_INFERENCE_STEPS   = 12                  # Standard FlowMatching, 12 steps
SLAT_CFG_STRENGTH      = 1.0                 # CFG strength for SLAT (from pipeline.yaml)
SLAT_RESCALE_T         = 1.0
SLAT_TIME_SCALE        = 1000.0

# ═══════════════════════════════════════════════════════════════════════════════
# SLAT Normalization Constants (from pipeline.yaml)
# ═══════════════════════════════════════════════════════════════════════════════
SLAT_MEAN = np.array([
     0.12211431,
     0.37204156,
    -1.26521907,
    -2.05276058,
    -3.10432536,
    -0.11294304,
    -0.85146744,
     0.45506954,
], dtype=np.float32)

SLAT_STD = np.array([
    2.37326008,
    2.13174402,
    2.2413953,
    2.30589401,
    2.1191894,
    1.8969511,
    2.41684989,
    2.08374642,
], dtype=np.float32)

# ═══════════════════════════════════════════════════════════════════════════════
# Sparse Conv Engine Constants
# ═══════════════════════════════════════════════════════════════════════════════
# SLAT I/O blocks use spconv3d.  Layout (io_block_channels=[128], num_io_res_blocks=2):
#   input_layer:      SparseLinear(8 → 128)
#   input_block[0]:   SparseResBlock3d(128→128, no downsample)
#                       conv1: SubMConv3d(128,128,3), conv2: SubMConv3d(128,128,3)
#   input_block[1]:   SparseResBlock3d(128→1024, downsample=stride-2)
#                       conv1: SubMConv3d(128,1024,3), conv2: SubMConv3d(1024,1024,3)
#                       downsample: SparseConv3d(128,1024,2,stride=2)  # or SparseDownsample
#                       skip: SparseLinear(128,1024)
#   [24 dense transformer blocks — no spconv]
#   out_block[0]:     SparseResBlock3d(1024*2→128, upsample=stride-2)  # skip cat doubles ch
#                       conv1: SubMConv3d(2048,128,3), conv2: SubMConv3d(128,128,3)
#                       upsample: SparseInverseConv3d(2048,128,2)
#                       skip: SparseLinear(2048,128)
#   out_block[1]:     SparseResBlock3d(128*2→128, no upsample)
#                       conv1: SubMConv3d(256,128,3), conv2: SubMConv3d(128,128,3)
#                       skip: SparseLinear(256,128)
#   out_layer:        SparseLinear(128 → 8)

SPARSE_CONV_KERNEL_SIZE     = 3              # 3×3×3 kernels
SPARSE_CONV_NUM_KERNEL_ELEM = 27             # 3³ = 27
SPARSE_CONV_HASH_SIZE       = 131072         # Hash table size (2^17)
SPARSE_CONV_MAX_NEIGHBORS   = 27             # Max neighbors per voxel

# ═══════════════════════════════════════════════════════════════════════════════
# Decoder Constants
# ═══════════════════════════════════════════════════════════════════════════════
GS_PARAM_DIM    = 14                         # per-voxel Gaussian params
MESH_SDF_DIM    = 1
MESH_DEFORM_DIM = 3

# ═══════════════════════════════════════════════════════════════════════════════
# Timestep Embedding
# ═══════════════════════════════════════════════════════════════════════════════
TIMESTEP_DIM = 256                           # Sinusoidal frequency embedding dimension
TIME_SCALE    = 1000.0


def prepare_timestep_schedule(
    n_steps: int,
    rescale_t: float = 1.0,
) -> np.ndarray:
    """Compute the rescaled timestep schedule for flow matching ODE.

    t_new = t / (1 + (rescale_t - 1) * (1 - t))

    With rescale_t=1.0 (default from pipeline.yaml) this is a linear schedule.
    """
    t_seq = np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float64)
    t_seq = t_seq / (1.0 + (rescale_t - 1.0) * (1.0 - t_seq))
    return t_seq.astype(np.float32)


def shortcut_step_sizes(n_steps: int) -> np.ndarray:
    """Returns the step sizes d for ShortCut ODE.

    ShortCut uses d = 1/current_resolution where resolutions double each step:
    With 2 steps and starting resolution=1: d = [1.0, 1.0] (full step both times
    in the simplified schedule used at inference_steps=2).

    The actual ShortCut schedule from shortcut/model.py:
      For inference_steps=2: ts = [0.0, 0.5, 1.0], ds = [0.5, 0.5]
    """
    ts = np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float32)
    ds = np.diff(ts)                         # step sizes: [0.5, 0.5] for n_steps=2
    return ts, ds
