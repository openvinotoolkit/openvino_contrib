/*
 * Copyright (C) 2018-2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */


// SAM 3D Objects — Sparse 3D Convolution OpenCL Kernels
//
// GPU kernels for sparse 3D convolution operations on Intel GPUs.
// Mirrors the BEVFusion sparse_conv.cl pattern with extensions for:
//   - SubMConv3d (submanifold convolution)
//   - Strided sparse conv (with int32 atomic scatter)
//   - Inverse conv (transpose upsample)
//   - Per-voxel linear transform
//   - Fused BN + ReLU
//   - Residual add + ReLU
//   - Sparse downsample (average pool)
//   - Sparse upsample (nearest neighbor)
//   - Sparse-to-dense scatter
//
// Kernel selection strategy:
//   C_in <= 256: slm_v2_w8  (full 27-neighbor SLM, 6–27 KB, good occupancy)
//   C_in >  256: slm_k1     (per-neighbor SLM, C_in*2 KB, full occupancy)
//     slm_v2_w8 for C_in=1024 needs 54 KB SLM → only ~1 WG/subslice (poor)
//     slm_k1    for C_in=1024 needs  2 KB SLM → ~32 WGs/subslice (full)
//
// All feature computation uses FP16 (half) for throughput;
// BN scale/bias remain FP32 for precision.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// ═══════════════════════════════════════════════════════════════════════════════
// Utility: FP32 ↔ FP16 conversion
// ═══════════════════════════════════════════════════════════════════════════════

__kernel void float_to_half_kernel(
    __global const float* restrict src,
    __global half* restrict dst,
    int total)
{
    int i = get_global_id(0);
    if (i < total) dst[i] = (half)src[i];
}

// ═══════════════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════════════
// Strided Conv (SparseConv3d stride=2) — FP16 with int32 atomic scatter
//
// For each input voxel, compute output = W @ feat and scatter-add to the
// output position determined by coord/stride. Uses int32 fixed-point atomics.
// ═══════════════════════════════════════════════════════════════════════════════

#define FP_SCALE 4096

__kernel void strided_conv_scatter_fp16(
    __global const half*  restrict feat_in,     // [N_in, C_in]
    __global const int*   restrict in_to_out,   // [N_in] — mapping to output idx
    __global const half*  restrict weights,     // [C_in * C_out] (center kernel only)
    __global const float* restrict scale,       // [C_out]
    __global const float* restrict bias,        // [C_out]
    int N_in, int C_in, int C_out,
    __global int* restrict out_accum,           // [N_out, C_out] int32 accumulator
    __global int* restrict out_counts,          // [N_out] int32 count
    int apply_bn)
{
    int i = get_global_id(0);
    if (i >= N_in) return;

    int out_idx = in_to_out[i];
    if (out_idx < 0) return;

    __global const half* f_i = feat_in + (long)i * C_in;

    for (int co = 0; co < C_out; co++) {
        float acc = 0.0f;
        for (int ci = 0; ci < C_in; ci++) {
            acc += (float)f_i[ci] * (float)weights[(long)ci * C_out + co];
        }
        int ival = (int)(acc * (float)FP_SCALE);
        atomic_add(&out_accum[(long)out_idx * C_out + co], ival);
    }
    atomic_add(&out_counts[out_idx], 1);
}

// Convert int32 fixed-point accumulation back to FP16 with averaging + BN
__kernel void strided_conv_gather_fp16(
    __global const int*   restrict accum,     // [N_out, C_out]
    __global const int*   restrict counts,    // [N_out]
    __global const float* restrict scale,     // [C_out]
    __global const float* restrict bias,      // [C_out]
    int N_out, int C_out,
    __global half* restrict feat_out,         // [N_out, C_out]
    int apply_bn)
{
    int j = get_global_id(0);
    int co = get_global_id(1);
    if (j >= N_out || co >= C_out) return;

    float val = (float)accum[(long)j * C_out + co] / (float)FP_SCALE;
    int cnt = counts[j];
    if (cnt > 1) val /= (float)cnt;

    if (apply_bn) val = val * scale[co] + bias[co];
    val = fmax(val, 0.0f);

    feat_out[(long)j * C_out + co] = (half)val;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Inverse Conv (SparseInverseConv3d) — for transposed upsampling
//
// Nearest-neighbor upsample + linear transform.
// ═══════════════════════════════════════════════════════════════════════════════

__kernel void inverse_conv_fp16(
    __global const half*  restrict feat_in,      // [N_coarse, C_in]
    __global const int*   restrict upsample_idx, // [N_fine] — maps fine→coarse
    __global const half*  restrict weights,      // [C_in * C_out]
    __global const float* restrict scale,        // [C_out]
    __global const float* restrict bias,         // [C_out]
    int N_fine, int C_in, int C_out,
    __global half* restrict feat_out,            // [N_fine, C_out]
    int apply_bn)
{
    int i = get_global_id(0);
    int ch_group = get_global_id(1);
    int ch_start = ch_group * 8;

    if (i >= N_fine || ch_start >= C_out) return;
    int ch_end = min(ch_start + 8, C_out);

    int src = upsample_idx[i];
    if (src < 0) {
        for (int co = ch_start; co < ch_end; co++)
            feat_out[(long)i * C_out + co] = (half)0.0f;
        return;
    }

    __global const half* f_src = feat_in + (long)src * C_in;

    for (int co = ch_start; co < ch_end; co++) {
        float acc = 0.0f;
        for (int ci = 0; ci < C_in; ci++) {
            acc += (float)f_src[ci] * (float)weights[(long)ci * C_out + co];
        }
        if (apply_bn) acc = acc * scale[co] + bias[co];
        acc = fmax(acc, 0.0f);
        feat_out[(long)i * C_out + co] = (half)acc;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Per-voxel Linear Transform (SparseLinear) — FP16
// ═══════════════════════════════════════════════════════════════════════════════

__kernel void linear_fp16(
    __global const half*  restrict feat_in,   // [N, C_in]
    __global const half*  restrict weights,   // [C_in * C_out]
    __global const float* restrict bias,      // [C_out] (may be zero)
    int N, int C_in, int C_out,
    __global half* restrict feat_out,         // [N, C_out]
    int has_bias)
{
    int i = get_global_id(0);
    int co = get_global_id(1);

    if (i >= N || co >= C_out) return;

    __global const half* f_i = feat_in + (long)i * C_in;
    float acc = has_bias ? bias[co] : 0.0f;

    for (int ci = 0; ci < C_in; ci++) {
        acc += (float)f_i[ci] * (float)weights[(long)ci * C_out + co];
    }

    feat_out[(long)i * C_out + co] = (half)acc;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Fused BN + ReLU (applied after conv when not already fused)
// ═══════════════════════════════════════════════════════════════════════════════

__kernel void apply_bn_relu(
    __global half* restrict feats,          // [N, C] — in-place
    __global const float* restrict scale,   // [C]
    __global const float* restrict bias,    // [C]
    int N, int C, int apply_relu)
{
    int i = get_global_id(0);
    int c = get_global_id(1);
    if (i >= N || c >= C) return;

    float val = (float)feats[(long)i * C + c];
    val = val * scale[c] + bias[c];
    if (apply_relu) val = fmax(val, 0.0f);
    feats[(long)i * C + c] = (half)val;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Residual Add + ReLU — FP16
// out[i] = max(0, a[i] + b[i])
// ═══════════════════════════════════════════════════════════════════════════════

__kernel void residual_add_relu_fp16(
    __global const half* restrict a,        // [N * C]
    __global const half* restrict b,        // [N * C]
    __global half* restrict out,            // [N * C]
    int total)
{
    int i = get_global_id(0);
    if (i >= total) return;
    float val = (float)a[i] + (float)b[i];
    out[i] = (half)fmax(val, 0.0f);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Downsample Average Pool — FP16
// Uses atomic int32 scatter-add (fixed-point) to average features.
// ═══════════════════════════════════════════════════════════════════════════════

__kernel void downsample_avg_fp16(
    __global const half*  restrict feat_in,   // [N_in, C]
    __global const int*   restrict in_to_out, // [N_in] mapping
    int N_in, int C,
    __global int* restrict out_accum,         // [N_out, C] int32
    __global int* restrict out_counts)        // [N_out]
{
    int i = get_global_id(0);
    if (i >= N_in) return;

    int out_idx = in_to_out[i];
    if (out_idx < 0) return;

    __global const half* f_i = feat_in + (long)i * C;
    for (int c = 0; c < C; c++) {
        int ival = (int)((float)f_i[c] * (float)FP_SCALE);
        atomic_add(&out_accum[(long)out_idx * C + c], ival);
    }
    atomic_add(&out_counts[out_idx], 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Upsample Nearest Neighbor — FP16
// ═══════════════════════════════════════════════════════════════════════════════

__kernel void upsample_nn_fp16(
    __global const half* restrict feat_in,      // [N_coarse, C]
    __global const int*  restrict upsample_idx, // [N_fine]
    int N_fine, int C,
    __global half* restrict feat_out)           // [N_fine, C]
{
    int i = get_global_id(0);
    int c = get_global_id(1);
    if (i >= N_fine || c >= C) return;

    int src = upsample_idx[i];
    feat_out[(long)i * C + c] = (src >= 0) ? feat_in[(long)src * C + c] : (half)0.0f;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Sparse-to-Dense scatter — FP16
// Scatters sparse features into a dense 3D grid.
// ═══════════════════════════════════════════════════════════════════════════════

__kernel void sparse_to_dense_fp16(
    __global const half*  restrict feat_in,   // [N, C]
    __global const int*   restrict coords,    // [N, 4] (batch, x, y, z)
    int N, int C,
    int grid_x, int grid_y, int grid_z,
    __global half* restrict dense_out)        // [B, C, X, Y, Z]
{
    int i = get_global_id(0);
    if (i >= N) return;

    int b = coords[i * 4 + 0];
    int x = coords[i * 4 + 1];
    int y = coords[i * 4 + 2];
    int z = coords[i * 4 + 3];

    if (x < 0 || x >= grid_x || y < 0 || y >= grid_y || z < 0 || z >= grid_z) return;

    long out_base = ((long)b * C * grid_x * grid_y * grid_z) +
                    ((long)x * grid_y * grid_z) +
                    ((long)y * grid_z) + z;

    __global const half* f_i = feat_in + (long)i * C;
    for (int c = 0; c < C; c++) {
        dense_out[out_base + (long)c * grid_x * grid_y * grid_z] = f_i[c];
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Gather features for one kernel position (for gather-GEMM-scatter sparse conv)
//
// For kernel position k_idx, gather features from neighbor[i, k_idx]:
//   gathered[i, :] = feat_in[nmap[i*27+k_idx], :]  if neighbor exists
//   gathered[i, :] = 0                               if neighbor absent (-1)
// ═══════════════════════════════════════════════════════════════════════════════

__kernel void gather_fp16(
    __global const half* restrict feat_in,   // [N, C]
    __global const int* restrict nmap,        // [N, 27]
    int N, int C, int k_idx,
    __global half* restrict gathered)          // [N, C]
{
    int i = get_global_id(0);    // voxel index
    int c = get_global_id(1);    // channel index
    if (i >= N || c >= C) return;

    int j = nmap[i * 27 + k_idx];
    gathered[(long)i * C + c] = (j >= 0) ? feat_in[(long)j * C + c] : (half)0.0f;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Scatter-add: output[i] += partial[i]  (element-wise, FP16)
// Used to accumulate partial products from 27 kernel positions.
// ═══════════════════════════════════════════════════════════════════════════════

__kernel void scatter_add_fp16(
    __global half* restrict output,           // [N, C_out] — accumulated
    __global const half* restrict partial,    // [N, C_out] — partial product
    int total)
{
    int i = get_global_id(0);
    if (i >= total)  return;
    output[i] = (half)((float)output[i] + (float)partial[i]);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Half-to-float conversion kernel (inverse of float_to_half_kernel)
// ═══════════════════════════════════════════════════════════════════════════════

__kernel void half_to_float_kernel(
    __global const half* restrict src,
    __global float* restrict dst,
    int total)
{
    int i = get_global_id(0);
    if (i < total) dst[i] = (float)src[i];
}

// ═══════════════════════════════════════════════════════════════════════════════
// SubMConv3d — SLM v2, 4 outputs/WI (ported from BEVFusion sparse_encoder)
//
// Key optimization: each work-group (= 1 voxel) cooperatively loads all 27
// neighbor feature rows into local memory ONCE, shared by all C_out work-items.
// This eliminates C_out-fold redundant global reads for the feature tensor.
//
// SLM requirement: 27 * C_in * sizeof(half) must be <= device local mem (~64KB)
// → Supports C_in up to 1213 channels.
//
// Dispatch: gws = {N * (C_out/4)}, lws = {C_out/4}  (1D, one WG per voxel)
// ═══════════════════════════════════════════════════════════════════════════════
__kernel void subm_conv_fp16_slm_v2(
    __global const half* input_features,     // [N, C_in]
    __global const int*  neighbor_map,       // [N, 27]
    __global const half* weights,            // [27, C_in, C_out]
    __global const float* bn_scale,          // [C_out]
    __global const float* bn_bias,           // [C_out]
    __global half* output_features,          // [N, C_out]
    const int N,
    const int C_in,
    const int C_out,
    const int apply_relu,
    __local half* local_feat                 // [27 * C_in]
) {
    int voxel_idx = get_group_id(0);
    int lid = get_local_id(0);
    int WG = get_local_size(0);

    if (voxel_idx >= N) return;

    // Preload neighbor indices into registers
    int nb[27];
    for (int k = 0; k < 27; k++) {
        nb[k] = neighbor_map[voxel_idx * 27 + k];
    }

    // Phase 1: Cooperatively load ALL neighbor features into SLM
    int total_load = 27 * C_in;
    for (int i = lid; i < total_load; i += WG) {
        int k = i / C_in;
        int c = i % C_in;
        local_feat[i] = (nb[k] >= 0) ? input_features[(long)nb[k] * C_in + c] : (half)0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: Compute 4 output channels from SLM + global weights
    int co = lid * 4;
    if (co >= C_out) return;

    float4 acc = (float4)(0.0f);

    for (int k = 0; k < 27; k++) {
        if (nb[k] < 0) continue;

        __local const half* f_row = local_feat + k * C_in;
        int wb = (int)((long)k * C_in * C_out + co);

        for (int c = 0; c < C_in; c++) {
            float fv = convert_float(f_row[c]);
            acc += fv * convert_float4(vload4(0, weights + wb + (long)c * C_out));
        }
    }

    // Fused BN + optional ReLU
    float4 sc = vload4(0, bn_scale + co);
    float4 bi = vload4(0, bn_bias + co);
    float4 v = acc * sc + bi;
    if (apply_relu) v = fmax(v, (float4)(0.0f));

    vstore4(convert_half4(v), 0, output_features + (long)voxel_idx * C_out + co);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SubMConv3d — SLM v2 w8, 8 outputs/WI (ported from BEVFusion sparse_encoder)
//
// Same as slm_v2 but processes 8 output channels per work-item using vload8/
// vstore8.  Better for C_out >= 64 (fewer work-items per voxel → lower sync
// overhead).
//
// Dispatch: gws = {N * (C_out/8)}, lws = {C_out/8}
// ═══════════════════════════════════════════════════════════════════════════════
__kernel void subm_conv_fp16_slm_v2_w8(
    __global const half* input_features,
    __global const int*  neighbor_map,
    __global const half* weights,
    __global const float* bn_scale,
    __global const float* bn_bias,
    __global half* output_features,
    const int N,
    const int C_in,
    const int C_out,
    const int apply_relu,
    __local half* local_feat             // [27 * C_in]
) {
    int voxel_idx = get_group_id(0);
    int lid = get_local_id(0);
    int WG = get_local_size(0);

    if (voxel_idx >= N) return;

    int nb[27];
    for (int k = 0; k < 27; k++) {
        nb[k] = neighbor_map[voxel_idx * 27 + k];
    }

    int total_load = 27 * C_in;
    for (int i = lid; i < total_load; i += WG) {
        int k = i / C_in;
        int c = i % C_in;
        local_feat[i] = (nb[k] >= 0) ? input_features[(long)nb[k] * C_in + c] : (half)0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int co = lid * 8;
    if (co >= C_out) return;

    float8 acc = (float8)(0.0f);

    for (int k = 0; k < 27; k++) {
        if (nb[k] < 0) continue;

        __local const half* f_row = local_feat + k * C_in;
        int wb = (int)((long)k * C_in * C_out + co);

        for (int c = 0; c < C_in; c++) {
            float fv = convert_float(f_row[c]);
            acc += fv * convert_float8(vload8(0, weights + wb + (long)c * C_out));
        }
    }

    float8 sc = vload8(0, bn_scale + co);
    float8 bi = vload8(0, bn_bias + co);
    float8 v = acc * sc + bi;
    if (apply_relu) v = fmax(v, (float8)(0.0f));

    vstore8(convert_half8(v), 0, output_features + (long)voxel_idx * C_out + co);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SubMConv3d — Tiled SLM (for C_in > 1213, up to C_in = 2048)
//
// Splits C_in into tiles of SLM_TILE_SIZE=1024.  Each tile loads 27×1024 half
// values (54 KB) into SLM, fits in Arc's 64 KB local memory.  For C_in=2048
// this runs 2 tile passes, reducing global feature reads by WG-fold vs no SLM.
//
// Dispatch: gws = {N * (C_out/8)}, lws = {C_out/8}
// SLM arg:  27 * 1024 * sizeof(half) = 55296 bytes (constant regardless of C_in)
// ═══════════════════════════════════════════════════════════════════════════════
#define SLM_TILE_SIZE 1024

__kernel void subm_conv_fp16_slm_tiled(
    __global const half* input_features,
    __global const int*  neighbor_map,
    __global const half* weights,
    __global const float* bn_scale,
    __global const float* bn_bias,
    __global half* output_features,
    const int N,
    const int C_in,
    const int C_out,
    const int apply_relu,
    __local half* local_feat             // [27 * SLM_TILE_SIZE]
) {
    int voxel_idx = get_group_id(0);
    int lid = get_local_id(0);
    int WG = get_local_size(0);

    if (voxel_idx >= N) return;

    int nb[27];
    for (int k = 0; k < 27; k++) {
        nb[k] = neighbor_map[voxel_idx * 27 + k];
    }

    int co = lid * 8;
    if (co >= C_out) return;

    float8 acc = (float8)(0.0f);

    int n_tiles = (C_in + SLM_TILE_SIZE - 1) / SLM_TILE_SIZE;
    for (int tile = 0; tile < n_tiles; tile++) {
        int tile_start = tile * SLM_TILE_SIZE;
        int tile_end = min(tile_start + SLM_TILE_SIZE, C_in);
        int tile_size = tile_end - tile_start;

        // Cooperatively load this tile of features for all 27 neighbors into SLM
        int total_load = 27 * tile_size;
        for (int i = lid; i < total_load; i += WG) {
            int k = i / tile_size;
            int c = i % tile_size;
            local_feat[k * SLM_TILE_SIZE + c] =
                (nb[k] >= 0) ? input_features[(long)nb[k] * C_in + tile_start + c] : (half)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute partial sums from SLM for this tile
        for (int k = 0; k < 27; k++) {
            if (nb[k] < 0) continue;
            __local const half* f_row = local_feat + k * SLM_TILE_SIZE;
            for (int c = 0; c < tile_size; c++) {
                float fv = convert_float(f_row[c]);
                long ci = tile_start + c;
                acc += fv * convert_float8(vload8(0, weights + (long)k * C_in * C_out + ci * C_out + co));
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float8 sc = vload8(0, bn_scale + co);
    float8 bi = vload8(0, bn_bias + co);
    float8 v = acc * sc + bi;
    if (apply_relu) v = fmax(v, (float8)(0.0f));

    vstore8(convert_half8(v), 0, output_features + (long)voxel_idx * C_out + co);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SubMConv3d — SLM_K1: per-neighbor SLM loading, one neighbor at a time.
//
// Key insight vs slm_v2_w8:
//   slm_v2_w8 loads ALL 27 neighbors into SLM = 27 * C_in * 2B
//     For C_in=1024: 54 KB — leaves room for only ~1 WG per 64 KB subslice
//     → ~32× lower occupancy → GPU cannot hide memory latency → slow.
//
//   slm_k1 loads ONE neighbor at a time into SLM = C_in * 2B
//     For C_in=1024: 2 KB — allows ~32 concurrent WGs per subslice
//     → Full occupancy → effective latency hiding → 5–15× faster on Arc.
//
// Trade-off: 27 barrier() calls instead of 2 in slm_v2_w8. Negligible at
// WG=128 threads because barrier cost is dwarfed by compute per iteration.
//
// Dispatch: GWS = N * (C_out/8),  LWS = C_out/8  (1D, one WG per voxel)
// SLM arg:  C_in * sizeof(half)   (2–4 KB for 1024–2048-channel cases)
// ═══════════════════════════════════════════════════════════════════════════════
// ─── slm_k1: SLM per-neighbor load, FP32 accumulation (reference path) ───────
// Use when C_out is too small for the f16 variant (C_out < 32 → WG < 4).
__kernel void subm_conv_fp16_slm_k1(
    __global const half* input_features,     // [N, C_in]
    __global const int*  neighbor_map,       // [N, 27]
    __global const half* weights,            // [27, C_in, C_out]
    __global const float* bn_scale,          // [C_out]
    __global const float* bn_bias,           // [C_out]
    __global half* output_features,          // [N, C_out]
    const int N,
    const int C_in,
    const int C_out,
    const int apply_relu,
    __local half* local_feat                 // [C_in]
) {
    int voxel_idx = get_group_id(0);
    int lid = get_local_id(0);
    int WG  = get_local_size(0);

    if (voxel_idx >= N) return;
    int co = lid * 8;
    if (co >= C_out) return;

    int nb[27];
    for (int k = 0; k < 27; k++)
        nb[k] = neighbor_map[voxel_idx * 27 + k];

    float8 acc = (float8)(0.0f);

    for (int k = 0; k < 27; k++) {
        if (nb[k] < 0) continue;
        __global const half* f_nb = input_features + (long)nb[k] * C_in;
        for (int ci = lid; ci < C_in; ci += WG)
            local_feat[ci] = f_nb[ci];
        barrier(CLK_LOCAL_MEM_FENCE);

        long wb = (long)k * C_in * C_out + co;
        for (int ci = 0; ci < C_in; ci++) {
            float fv = convert_float(local_feat[ci]);
            acc += fv * convert_float8(vload8(0, weights + wb + (long)ci * C_out));
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float8 sc = vload8(0, bn_scale + co);
    float8 bi = vload8(0, bn_bias + co);
    float8 v = acc * sc + bi;
    if (apply_relu) v = fmax(v, (float8)(0.0f));
    vstore8(convert_half8(v), 0, output_features + (long)voxel_idx * C_out + co);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SubMConv3d — SLM_K1_F16: SLM per-neighbor load + FP16 inner accumulation.
//
// Combines the SLM input-feature sharing of slm_k1 (input read reduction
// from 128× duplication to 1×) with the 2× FP16 compute throughput of lite8
// (half8 accumulation instead of float8).
//
// Key improvements over slm_k1 (FP32):
//  1. half8 hacc instead of float8 acc in the inner ci loop → 2× FLOPs/cycle
//  2. 4-channel inner loop unrolling → exposes ILP to hide weight-load latency
//  3. Per-neighbor flush to FP32 (acc += convert_float8(hacc)) → no overflow
//
// Safe for typical normalized feature values: per-neighbor sum over C_in=1024
// halfs ≈ sqrt(1024) ≈ 32 in magnitude, far below FP16 max (65504).
//
// Dispatch: GWS = N * (C_out/8), LWS = C_out/8
// SLM:     C_in * sizeof(half) (2–4 KB for 1024–2048 ch)
// ═══════════════════════════════════════════════════════════════════════════════
__kernel void subm_conv_fp16_slm_k1_f16(
    __global const half* input_features,
    __global const int*  neighbor_map,
    __global const half* weights,
    __global const float* bn_scale,
    __global const float* bn_bias,
    __global half* output_features,
    const int N,
    const int C_in,
    const int C_out,
    const int apply_relu,
    __local half* local_feat                 // [C_in]
) {
    int voxel_idx = get_group_id(0);
    int lid = get_local_id(0);
    int WG  = get_local_size(0);

    if (voxel_idx >= N) return;
    int co = lid * 8;
    if (co >= C_out) return;

    int nb[27];
    for (int k = 0; k < 27; k++)
        nb[k] = neighbor_map[voxel_idx * 27 + k];

    float8 acc = (float8)(0.0f);

    for (int k = 0; k < 27; k++) {
        if (nb[k] < 0) continue;

        // Phase 1: cooperatively load ONE neighbor into SLM (C_in halfs = 2-4 KB)
        __global const half* f_nb = input_features + (long)nb[k] * C_in;
        for (int ci = lid; ci < C_in; ci += WG)
            local_feat[ci] = f_nb[ci];
        barrier(CLK_LOCAL_MEM_FENCE);

        // Phase 2: FP16 inner accumulation with 4-channel unrolling.
        // Each of the 4 independent hacc ops can be pipelined by the GPU
        // scheduler, hiding the memory-load latency for weight vload8s.
        long wb = (long)k * C_in * C_out + co;
        half8 hacc = (half8)(0.0h);
        int ci = 0;
        for (; ci + 3 < C_in; ci += 4) {
            hacc += local_feat[ci]   * vload8(0, weights + wb + (long)ci     * C_out);
            hacc += local_feat[ci+1] * vload8(0, weights + wb + (long)(ci+1) * C_out);
            hacc += local_feat[ci+2] * vload8(0, weights + wb + (long)(ci+2) * C_out);
            hacc += local_feat[ci+3] * vload8(0, weights + wb + (long)(ci+3) * C_out);
        }
        for (; ci < C_in; ci++)
            hacc += local_feat[ci] * vload8(0, weights + wb + (long)ci * C_out);

        acc += convert_float8(hacc);  // flush per-neighbor to prevent FP16 overflow
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float8 sc = vload8(0, bn_scale + co);
    float8 bi = vload8(0, bn_bias + co);
    float8 v = acc * sc + bi;
    if (apply_relu) v = fmax(v, (float8)(0.0f));
    vstore8(convert_half8(v), 0, output_features + (long)voxel_idx * C_out + co);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SubMConv3d — SLM_K1_INT8: INT8 weight compression + SLM feature caching.
//
// Same structure as slm_k1_f16 but weights are stored as INT8 (1 byte each)
// instead of FP16 (2 bytes), halving weight bandwidth for BW-bound convolutions.
//
// Quantization: per-kernel-position per-output-channel symmetric INT8.
//   w_f16 = convert_half8(w_int8[k*C_in*C_out + ci*C_out + co : co+8]) * wscale[k*C_out+co : co+8]
//   scale[k][co] = max|w[k, :, co]| / 127, stored in weights_quant_scale as FP16
//
// The quantization scale is applied ONCE per k-neighbor (not per ci), so the
// only extra cost over slm_k1_f16 is one vload8 of scales per k (negligible).
//
// Weight bandwidth: C_in × 1B (INT8) instead of C_in × 2B (FP16) → 2× less
// For weight-BW-bound ops (1024→1024, 2048→128): expected ~1.5-2× speedup.
//
// Accuracy: per-k per-co INT8 symmetric quantization gives <1% relative error
// for well-trained FP16 models with diverse weight distributions.
//
// Dispatch: same as slm_k1_f16 — GWS = N*(C_out/8), LWS = C_out/8
// SLM:     C_in * sizeof(half) (per-neighbor, same as slm_k1_f16)
// ═══════════════════════════════════════════════════════════════════════════════
__kernel void subm_conv_fp16_slm_k1_int8(
    __global const half* input_features,
    __global const int*  neighbor_map,
    __global const char* weights_int8,       // [27 * C_in * C_out] INT8 (1B each)
    __global const half* weights_quant_scale,// [27 * C_out] FP16 — per-k, per-out-ch scale
    __global const float* bn_scale,
    __global const float* bn_bias,
    __global half* output_features,
    const int N,
    const int C_in,
    const int C_out,
    const int apply_relu,
    __local half* local_feat                 // [C_in]
) {
    int voxel_idx = get_group_id(0);
    int lid = get_local_id(0);
    int WG  = get_local_size(0);

    if (voxel_idx >= N) return;
    int co = lid * 8;
    if (co >= C_out) return;

    int nb[27];
    for (int k = 0; k < 27; k++)
        nb[k] = neighbor_map[voxel_idx * 27 + k];

    float8 acc = (float8)(0.0f);

    for (int k = 0; k < 27; k++) {
        if (nb[k] < 0) continue;

        // Phase 1: cooperatively load ONE neighbor's FP16 features into SLM
        __global const half* f_nb = input_features + (long)nb[k] * C_in;
        for (int ci = lid; ci < C_in; ci += WG)
            local_feat[ci] = f_nb[ci];
        barrier(CLK_LOCAL_MEM_FENCE);

        // Phase 2: INT8 weight loads with per-k,per-co dequantization.
        // Load 8 INT8 weights (8 bytes, one char8), convert to half8, multiply
        // by per-output-channel scale (one vload8 of FP16 scales), then MAD.
        // The scale load is hoisted outside the ci loop: one vload8 per k = cheap.
        long wb    = (long)k * C_in * C_out + co;   // byte offset = half offset (INT8)
        long ws    = (long)k * C_out + co;           // scale offset
        half8 wscale = vload8(0, weights_quant_scale + ws);  // [co..co+7] scales

        // Accumulate unscaled INT8-converted weights, apply scale once at end of k.
        // This avoids C_in multiplications by wscale (replaced by 1 per k).
        half8 hacc = (half8)(0.0h);
        int ci = 0;
        for (; ci + 3 < C_in; ci += 4) {
            // Load 4 × 8-byte weight vectors (4 char8s = 32 bytes total)
            char8 wi0 = vload8(0, weights_int8 + wb + (long)ci     * C_out);
            char8 wi1 = vload8(0, weights_int8 + wb + (long)(ci+1) * C_out);
            char8 wi2 = vload8(0, weights_int8 + wb + (long)(ci+2) * C_out);
            char8 wi3 = vload8(0, weights_int8 + wb + (long)(ci+3) * C_out);
            hacc += local_feat[ci]   * convert_half8(wi0);
            hacc += local_feat[ci+1] * convert_half8(wi1);
            hacc += local_feat[ci+2] * convert_half8(wi2);
            hacc += local_feat[ci+3] * convert_half8(wi3);
        }
        for (; ci < C_in; ci++) {
            char8 wi = vload8(0, weights_int8 + wb + (long)ci * C_out);
            hacc += local_feat[ci] * convert_half8(wi);
        }

        // Apply quantization scale and flush to FP32 (prevents FP16 overflow)
        acc += convert_float8(hacc * wscale);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float8 sc = vload8(0, bn_scale + co);
    float8 bi = vload8(0, bn_bias + co);
    float8 v = acc * sc + bi;
    if (apply_relu) v = fmax(v, (float8)(0.0f));
    vstore8(convert_half8(v), 0, output_features + (long)voxel_idx * C_out + co);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SubMConv3d — SLM_MV: Multi-Voxel weight sharing with SLM feature caching.
//
// The 1024→1024 conv is weight-bandwidth-bound: 56 MB weights × 27 neighbors
// needs 1.5 GB of weight reads per step. Processing NV=4 voxels per WG means
// each weight element is accessed once per WG but reused for 4 voxels, reducing
// effective weight bandwidth pressure by 4× through L2/L3 cache sharing.
//
// WG layout: NV voxels × (C_out/8) output tiles = NV*(C_out/8) work-items
//   WG = 4 * 128 = 512 for 1024-channel
//   - WIs 0..C_out/8-1 handle voxel 0
//   - WIs C_out/8..2*C_out/8-1 handle voxel 1
//   - etc.
//
// SLM: NV * C_in * sizeof(half) = 4 * 1024 * 2 = 8 KB
// Weight reads: 27 * C_in * C_out * 2B shared across NV voxels
// Occupancy: 64KB / 8KB = 8 WGs/subslice
//
// Register layout (per thread): nb[27] = 27 ints (for this thread's voxel only).
// This is 108B vs the old nb[NV][27]=432B design which caused register spilling.
//
// Dispatch: GWS = ceil(N/NV) * WG, LWS = WG
// ═══════════════════════════════════════════════════════════════════════════════
#define MV_NV 4
__kernel void subm_conv_fp16_slm_mv(
    __global const half* input_features,
    __global const int*  neighbor_map,
    __global const half* weights,
    __global const float* bn_scale,
    __global const float* bn_bias,
    __global half* output_features,
    const int N,
    const int C_in,
    const int C_out,
    const int apply_relu,
    __local half* local_feat                 // [MV_NV * C_in]
) {
    int group_id  = get_group_id(0);
    int lid       = get_local_id(0);
    int WG        = get_local_size(0);       // = MV_NV * (C_out / 8)
    int co_stride = C_out / 8;              // work-items per voxel

    // Which voxel within the group, and which output tile within that voxel
    int local_vi  = lid / co_stride;
    int local_coi = lid % co_stride;
    int voxel_idx = group_id * MV_NV + local_vi;
    int co        = local_coi * 8;

    // Each thread pre-loads the 27 neighbor indices for ITS OWN voxel only.
    // This uses 27 ints (108B) instead of the old nb[MV_NV][27] (108 ints = 432B)
    // which caused register spilling and performance regression.
    int nb[27];
    for (int k = 0; k < 27; k++)
        nb[k] = (voxel_idx < N) ? neighbor_map[voxel_idx * 27 + k] : -1;

    float8 acc = (float8)(0.0f);

    for (int k = 0; k < 27; k++) {
        // Phase 1: cooperatively load all NV voxels' features for neighbor k
        // into SLM. Each WI covers elements [lid, lid+WG, lid+2*WG, ...] of
        // the MV_NV*C_in flat SLM array. We re-read neighbor_map directly here
        // (neighbor_map is small ~1.3MB and cache-resident) to avoid the
        // nb[MV_NV][27] register array.
        int total_load = MV_NV * C_in;
        for (int i = lid; i < total_load; i += WG) {
            int vi = i / C_in;
            int ci = i % C_in;
            int gvi = group_id * MV_NV + vi;
            int n_idx = (gvi < N) ? neighbor_map[gvi * 27 + k] : -1;
            local_feat[vi * C_in + ci] = (n_idx >= 0)
                ? input_features[(long)n_idx * C_in + ci] : (half)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Phase 2: compute using weight tile shared across all NV voxels.
        // Each WI computes 8 output channels for ITS voxel using SLM features.
        // The weight vload8 at the same co is shared (L2 cache hit) across all
        // NV voxels with the same co_idx, reducing effective weight bandwidth.
        if (voxel_idx < N && co < C_out && nb[k] >= 0) {
            long wb = (long)k * C_in * C_out + co;
            __local const half* f_row = local_feat + local_vi * C_in;
            half8 hacc = (half8)(0.0h);
            int ci = 0;
            for (; ci + 3 < C_in; ci += 4) {
                hacc += f_row[ci]   * vload8(0, weights + wb + (long)ci     * C_out);
                hacc += f_row[ci+1] * vload8(0, weights + wb + (long)(ci+1) * C_out);
                hacc += f_row[ci+2] * vload8(0, weights + wb + (long)(ci+2) * C_out);
                hacc += f_row[ci+3] * vload8(0, weights + wb + (long)(ci+3) * C_out);
            }
            for (; ci < C_in; ci++)
                hacc += f_row[ci] * vload8(0, weights + wb + (long)ci * C_out);
            acc += convert_float8(hacc);
        }
        // All threads must reach this barrier before next k's Phase 1 writes SLM.
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (voxel_idx < N && co < C_out) {
        float8 sc = vload8(0, bn_scale + co);
        float8 bi = vload8(0, bn_bias + co);
        float8 v  = acc * sc + bi;
        if (apply_relu) v = fmax(v, (float8)(0.0f));
        vstore8(convert_half8(v), 0, output_features + (long)voxel_idx * C_out + co);
    }
}
#undef MV_NV

// ═══════════════════════════════════════════════════════════════════════════════
// SubMConv3d — SLM_VTILE: in-register weight reuse across a tile of VT voxels.
//
// Unlike slm_mv (which assigned one thread per (voxel, co) and relied on L2 to
// share weight reads across voxels), here a SINGLE thread computes the SAME 8
// output channels for VT_TV voxels. Each weight vload8 is issued ONCE and reused
// across VT_TV register accumulators → genuine VT_TV× weight-bandwidth reduction
// (not L2-dependent). This directly attacks the weight-BW-bound convs
// (out.0 2048→128, out.1 256→128, in.1 conv2 1024→1024).
//
// Thread/WG layout:
//   WG processes VT_TV consecutive voxels. WG size = C_out/8 threads.
//   Thread `lid` owns output-channel tile co = lid*8 for ALL VT_TV voxels.
// SLM: VT_TV * C_in * sizeof(half)  (one neighbor's features for all VT_TV voxels)
// Dispatch: GWS = ceil(N/VT_TV) * (C_out/8),  LWS = C_out/8
//
// FP16 accumulation flushed to FP32 per-neighbor (same overflow safety as
// slm_k1_f16): per-neighbor partial sum over C_in halfs stays well below FP16 max.
// ═══════════════════════════════════════════════════════════════════════════════
#define VT_TV 4
__kernel void subm_conv_fp16_slm_vtile(
    __global const half* input_features,
    __global const int*  neighbor_map,
    __global const half* weights,
    __global const float* bn_scale,
    __global const float* bn_bias,
    __global half* output_features,
    const int N,
    const int C_in,
    const int C_out,
    const int apply_relu,
    __local half* local_feat                 // [VT_TV * C_in]
) {
    int group_id = get_group_id(0);
    int lid      = get_local_id(0);
    int WG       = get_local_size(0);        // = C_out / 8
    int co       = lid * 8;
    int base_vox = group_id * VT_TV;

    float8 acc0 = (float8)(0.0f);
    float8 acc1 = (float8)(0.0f);
    float8 acc2 = (float8)(0.0f);
    float8 acc3 = (float8)(0.0f);

    for (int k = 0; k < 27; k++) {
        // Phase 1: cooperatively load VT_TV voxels' neighbor-k features into SLM.
        int total_load = VT_TV * C_in;
        for (int i = lid; i < total_load; i += WG) {
            int vi = i / C_in;
            int ci = i % C_in;
            int gv = base_vox + vi;
            int n_idx = (gv < N) ? neighbor_map[gv * 27 + k] : -1;
            local_feat[vi * C_in + ci] =
                (n_idx >= 0) ? input_features[(long)n_idx * C_in + ci] : (half)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Phase 2: each thread reuses one weight vload8 across all VT_TV voxels.
        if (co < C_out) {
            long wb = (long)k * C_in * C_out + co;
            __local const half* f0 = local_feat + 0 * C_in;
            __local const half* f1 = local_feat + 1 * C_in;
            __local const half* f2 = local_feat + 2 * C_in;
            __local const half* f3 = local_feat + 3 * C_in;
            half8 h0 = (half8)(0.0h), h1 = (half8)(0.0h);
            half8 h2 = (half8)(0.0h), h3 = (half8)(0.0h);
            for (int ci = 0; ci < C_in; ci++) {
                half8 w = vload8(0, weights + wb + (long)ci * C_out);
                h0 += f0[ci] * w;
                h1 += f1[ci] * w;
                h2 += f2[ci] * w;
                h3 += f3[ci] * w;
            }
            acc0 += convert_float8(h0);
            acc1 += convert_float8(h1);
            acc2 += convert_float8(h2);
            acc3 += convert_float8(h3);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (co < C_out) {
        float8 sc = vload8(0, bn_scale + co);
        float8 bi = vload8(0, bn_bias + co);
        int gv0 = base_vox + 0, gv1 = base_vox + 1;
        int gv2 = base_vox + 2, gv3 = base_vox + 3;
        if (gv0 < N) { float8 v = acc0 * sc + bi; if (apply_relu) v = fmax(v, (float8)(0.0f)); vstore8(convert_half8(v), 0, output_features + (long)gv0 * C_out + co); }
        if (gv1 < N) { float8 v = acc1 * sc + bi; if (apply_relu) v = fmax(v, (float8)(0.0f)); vstore8(convert_half8(v), 0, output_features + (long)gv1 * C_out + co); }
        if (gv2 < N) { float8 v = acc2 * sc + bi; if (apply_relu) v = fmax(v, (float8)(0.0f)); vstore8(convert_half8(v), 0, output_features + (long)gv2 * C_out + co); }
        if (gv3 < N) { float8 v = acc3 * sc + bi; if (apply_relu) v = fmax(v, (float8)(0.0f)); vstore8(convert_half8(v), 0, output_features + (long)gv3 * C_out + co); }
    }
}
#undef VT_TV

// ═══════════════════════════════════════════════════════════════════════════════
// Affine LayerNorm + SiLU — FP16 in/out, FP32 gamma/beta
// One work-item per row (voxel). For C <= 2048 this is efficient.
// norm1: h = silu(layernorm(x, gamma, beta))
// ═══════════════════════════════════════════════════════════════════════════════

__kernel void layernorm_silu_fp16(
    __global const half* restrict feat_in,    // [N, C]
    __global const float* restrict gamma,     // [C]
    __global const float* restrict beta,      // [C]
    int N, int C, float eps,
    __global half* restrict feat_out)         // [N, C]
{
    int i = get_global_id(0);
    if (i >= N) return;

    __global const half* row = feat_in + (long)i * C;

    // Compute mean
    float sum = 0.0f;
    for (int c = 0; c < C; c++) sum += (float)row[c];
    float mean = sum / (float)C;

    // Compute variance
    float var_sum = 0.0f;
    for (int c = 0; c < C; c++) {
        float d = (float)row[c] - mean;
        var_sum += d * d;
    }
    float inv_std = rsqrt(var_sum / (float)C + eps);

    // Normalize + affine + SiLU
    __global half* out_row = feat_out + (long)i * C;
    for (int c = 0; c < C; c++) {
        float x = ((float)row[c] - mean) * inv_std;
        float y = x * gamma[c] + beta[c];
        // SiLU = y * sigmoid(y)
        float sig = 1.0f / (1.0f + exp(-y));
        out_row[c] = (half)(y * sig);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Non-affine LayerNorm + emb_scale/shift + SiLU — FP16 in/out
// norm2_act: h = silu(layernorm(x) * (1 + scale) + shift)
// scale/shift are per-channel from timestep embedding (broadcast over N)
// ═══════════════════════════════════════════════════════════════════════════════

__kernel void layernorm_scale_shift_silu_fp16(
    __global const half* restrict feat_in,    // [N, C]
    __global const float* restrict scale,     // [C] — emb_scale
    __global const float* restrict shift,     // [C] — emb_shift
    int N, int C, float eps,
    __global half* restrict feat_out)         // [N, C]
{
    int i = get_global_id(0);
    if (i >= N) return;

    __global const half* row = feat_in + (long)i * C;

    // Compute mean
    float sum = 0.0f;
    for (int c = 0; c < C; c++) sum += (float)row[c];
    float mean = sum / (float)C;

    // Compute variance
    float var_sum = 0.0f;
    for (int c = 0; c < C; c++) {
        float d = (float)row[c] - mean;
        var_sum += d * d;
    }
    float inv_std = rsqrt(var_sum / (float)C + eps);

    // Normalize + scale/shift + SiLU
    __global half* out_row = feat_out + (long)i * C;
    for (int c = 0; c < C; c++) {
        float x = ((float)row[c] - mean) * inv_std;
        float y = x * (1.0f + scale[c]) + shift[c];
        float sig = 1.0f / (1.0f + exp(-y));
        out_row[c] = (half)(y * sig);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Residual Add (no activation) — FP16
// out = a + b (used for skip+conv output)
// ═══════════════════════════════════════════════════════════════════════════════

__kernel void residual_add_fp16(
    __global const half* restrict a,          // [total]
    __global const half* restrict b,          // [total]
    __global half* restrict out,              // [total]
    int total)
{
    int i = get_global_id(0);
    if (i >= total) return;
    out[i] = (half)((float)a[i] + (float)b[i]);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Copy FP16 buffer (for saving identity before norm1 modifies it)
// ═══════════════════════════════════════════════════════════════════════════════

__kernel void copy_fp16(
    __global const half* restrict src,
    __global half* restrict dst,
    int total)
{
    int i = get_global_id(0);
    if (i >= total) return;
    dst[i] = src[i];
}

// ═══════════════════════════════════════════════════════════════════════════════
// FP32 MESH UPSAMPLE PATH (SparseMeshUpsample op)
// Self-contained, accuracy-critical (feeds FlexiCubes SDF). All FP32.
// Convention matches CPU reference (mesh_extract_ref.py) exactly.
// ═══════════════════════════════════════════════════════════════════════════════

// GroupNorm32 reduction: per-channel sum and sumsq over all N voxels (FP32).
// One work-item per channel c; loops over N (column access). C is small (<=768).
__kernel void gn_reduce_fp32(
    __global const float* restrict feat,   // [N, C]
    int N, int C,
    __global float* restrict out_sum,      // [C]
    __global float* restrict out_sumsq)    // [C]
{
    int c = get_global_id(0);
    if (c >= C) return;
    float s = 0.0f, sq = 0.0f;
    for (int n = 0; n < N; n++) {
        float v = feat[(long)n * C + c];
        s += v;
        sq += v * v;
    }
    out_sum[c]   = s;
    out_sumsq[c] = sq;
}

// GroupNorm reduce pass 1, PARTIAL/parallel: grid [C, P]. Each (c,p) sums the
// strided partition n = p, p+P, p+2P, ... into partial buffers [C*P]. A single
// C-thread serial reduction over all N would trip the GPU execution watchdog for
// large N (e.g. block1 gn1: N=580736), so the work is spread over P partitions.
__kernel void gn_partial_fp32(
    __global const float* restrict feat,   // [N, C]
    int N, int C, int P,
    __global float* restrict part_sum,     // [C, P]
    __global float* restrict part_sumsq)   // [C, P]
{
    int c = get_global_id(0);
    int p = get_global_id(1);
    if (c >= C || p >= P) return;
    float s = 0.0f, sq = 0.0f;
    for (int n = p; n < N; n += P) {
        float v = feat[(long)n * C + c];
        s += v;
        sq += v * v;
    }
    part_sum[(long)c * P + p]   = s;
    part_sumsq[(long)c * P + p] = sq;
}

// GroupNorm reduce, pass 2: sum of squared deviations from per-channel mean.
__kernel void gn_reduce_sqdev_fp32(
    __global const float* restrict feat,   // [N, C]
    __global const float* restrict mean,   // [C] (per-channel, expanded from group)
    int N, int C,
    __global float* restrict out_sqdev)    // [C]
{
    int c = get_global_id(0);
    if (c >= C) return;
    float m = mean[c];
    float sq = 0.0f;
    for (int n = 0; n < N; n++) {
        float d = feat[(long)n * C + c] - m;
        sq += d * d;
    }
    out_sqdev[c] = sq;
}

// GroupNorm reduce pass 2, PARTIAL/parallel: grid [C, P]. Sum of squared
// deviations from per-channel mean, spread over P partitions (see gn_partial_fp32).
__kernel void gn_partial_sqdev_fp32(
    __global const float* restrict feat,   // [N, C]
    __global const float* restrict mean,   // [C]
    int N, int C, int P,
    __global float* restrict part_sqdev)   // [C, P]
{
    int c = get_global_id(0);
    int p = get_global_id(1);
    if (c >= C || p >= P) return;
    float m = mean[c];
    float sq = 0.0f;
    for (int n = p; n < N; n += P) {
        float d = feat[(long)n * C + c] - m;
        sq += d * d;
    }
    part_sqdev[(long)c * P + p] = sq;
}

// GroupNorm32 apply + affine + optional SiLU (FP32).
// mean/inv_std are per-channel (expanded from per-group values by host).
__kernel void gn_affine_silu_fp32(
    __global const float* restrict feat_in,  // [N, C]
    __global const float* restrict mean,     // [C]
    __global const float* restrict inv_std,  // [C]
    __global const float* restrict gamma,    // [C]
    __global const float* restrict beta,     // [C]
    int N, int C, int apply_silu,
    __global float* restrict feat_out)       // [N, C]
{
    int n = get_global_id(0);
    int c = get_global_id(1);
    if (n >= N || c >= C) return;
    long idx = (long)n * C + c;
    float x = (feat_in[idx] - mean[c]) * inv_std[c];
    float y = x * gamma[c] + beta[c];
    if (apply_silu) {
        float sig = 1.0f / (1.0f + exp(-y));
        y = y * sig;
    }
    feat_out[idx] = y;
}

// Submanifold conv3d (stride=1), FP32, NO ReLU.
// weight layout [Cout, 27, Cin] = raw reshape of checkpoint (Cout,kz,ky,kx,Cin).
// nmap[p*27+k] = index of neighbor at offset (kz-1,ky-1,kx-1), k=kz*9+ky*3+kx.
//
// row_shift folds a subdivide-by-8 gather into the read: with row_shift=3 the
// kernel reads the COARSE feature buffer directly (row j>>3) instead of a
// materialised [8N, Cin] copy, which for the mesh decoder saves a 1.6 GB
// device allocation and the bandwidth to write and re-read it.  row_shift=0
// is the plain case.
__kernel void subm_conv3d_fp32(
    __global const float* restrict feat_in,  // [N >> row_shift, Cin]
    __global const float* restrict weight,   // [Cout*27*Cin]
    __global const float* restrict bias,     // [Cout]
    __global const int* restrict nmap,       // [N, 27]
    int N, int Cin, int Cout, int p_off, int row_shift,
    __global float* restrict feat_out)       // [N, Cout]
{
    int p  = get_global_id(0) + p_off;
    int co = get_global_id(1);
    if (p >= N || co >= Cout) return;
    __global const int* nb = nmap + (long)p * 27;
    __global const float* w_co = weight + (long)co * 27 * Cin;
    float acc = bias[co];
    for (int k = 0; k < 27; k++) {
        int j = nb[k];
        if (j < 0) continue;
        __global const float* frow = feat_in + (long)(j >> row_shift) * Cin;
        __global const float* wk = w_co + (long)k * Cin;
        for (int ci = 0; ci < Cin; ci++) {
            acc += frow[ci] * wk[ci];
        }
    }
    feat_out[(long)p * Cout + co] = acc;
}

// Register-tiled submanifold conv3d, FP32.  Numerically identical to
// subm_conv3d_fp32 (same FMA order over k then ci) but ~5x less global
// traffic.
//
// weight_t layout is [27, Cin, Cout] -- transposed on the host from the
// checkpoint's [Cout, 27, Cin] -- so the MESH_TCO output channels a work-item
// owns are contiguous, making the weight read fully coalesced across the
// subgroup.
//
// Each work-item computes MESH_TP voxels x MESH_TCO output channels with the
// accumulators held in registers.  Per inner iteration it loads MESH_TP
// feature scalars + MESH_TCO weights and issues MESH_TP*MESH_TCO FMAs, i.e.
// ~5 FLOP per loaded float versus ~0.5 for the scalar kernel above.
//
// Requires Cout % MESH_TCO == 0 (192 and 96 in the mesh decoder); the caller
// falls back to subm_conv3d_fp32 otherwise.
// Tile sizes: MESH_TP*MESH_TCO accumulators are held in registers, so keeping
// the product at 32 floats is what matters -- 4x16 was measured 3x SLOWER than
// 4x8 on Xe2 because 64 accumulators spill.
#define MESH_TP  4
#define MESH_TCO 8

__kernel void subm_conv3d_fp32_tiled(
    __global const float* restrict feat_in,   // [N >> row_shift, Cin]
    __global const float* restrict weight_t,  // [27, Cin, Cout]
    __global const float* restrict bias,      // [Cout]
    __global const int* restrict nmap,        // [N, 27]
    int N, int Cin, int Cout, int p_off, int n_chunk, int row_shift,
    __global float* restrict feat_out)        // [N, Cout]
{
    int co_base = (int)get_global_id(0) * MESH_TCO;
    int p_base  = p_off + (int)get_global_id(1) * MESH_TP;
    int p_end   = min(p_off + n_chunk, N);
    if (co_base >= Cout || p_base >= p_end) return;

    float8 acc[MESH_TP];
    float8 b = vload8(0, bias + co_base);
    #pragma unroll
    for (int t = 0; t < MESH_TP; t++) acc[t] = b;

    for (int k = 0; k < 27; k++) {
        // Missing neighbours are handled branchlessly: clamp the row index to 0
        // (always valid) and zero the contribution via the mask.
        int   j[MESH_TP];
        float m[MESH_TP];
        #pragma unroll
        for (int t = 0; t < MESH_TP; t++) {
            int p = p_base + t;
            int jj = (p < p_end) ? nmap[(long)p * 27 + k] : -1;
            j[t] = jj > 0 ? (jj >> row_shift) : 0;
            m[t] = jj >= 0 ? 1.0f : 0.0f;
        }
        __global const float* wk = weight_t + (long)k * Cin * Cout + co_base;
        for (int ci = 0; ci < Cin; ci++) {
            float8 w = vload8(0, wk + (long)ci * Cout);
            #pragma unroll
            for (int t = 0; t < MESH_TP; t++) {
                float f = feat_in[(long)j[t] * Cin + ci] * m[t];
                acc[t] = fma((float8)f, w, acc[t]);
            }
        }
    }

    #pragma unroll
    for (int t = 0; t < MESH_TP; t++) {
        int p = p_base + t;
        if (p < p_end) vstore8(acc[t], 0, feat_out + (long)p * Cout + co_base);
    }
}

// Per-voxel linear (used for k1 skip conv and out_layer). weight [Cout, Cin].
// row_shift folds a subdivide-by-8 gather into the read (see subm_conv3d_fp32).
// accumulate=1 does feat_out += ... instead of feat_out = ..., which lets the
// skip connection land straight on the conv output and skips a separate add
// pass (and its full-size destination buffer).
__kernel void dense_linear_fp32(
    __global const float* restrict feat_in,  // [N >> row_shift, Cin]
    __global const float* restrict weight,   // [Cout, Cin]
    __global const float* restrict bias,     // [Cout]
    int N, int Cin, int Cout,
    __global float* restrict feat_out,       // [N, Cout]
    int p_off, int n_chunk, int row_shift, int accumulate)
{
    long gid = get_global_id(0);
    long total = (long)n_chunk * Cout;
    if (gid >= total) return;
    int p  = p_off + (int)(gid / Cout);
    int co = (int)(gid % Cout);
    __global const float* frow = feat_in + (long)(p >> row_shift) * Cin;
    __global const float* wrow = weight + (long)co * Cin;
    float acc = bias[co];
    for (int ci = 0; ci < Cin; ci++) acc += frow[ci] * wrow[ci];
    long o = (long)p * Cout + co;
    feat_out[o] = accumulate ? (feat_out[o] + acc) : acc;
    (void)N;
}

// Subdivide: tile each voxel's features to 8 children. feat_out[n*8+child] = feat_in[n].
__kernel void subdivide_tile_fp32(
    __global const float* restrict feat_in,  // [N, C]
    int N, int C,
    __global float* restrict feat_out)       // [8N, C]
{
    int pout = get_global_id(0);   // [0, 8N)
    int c    = get_global_id(1);
    if (pout >= N * 8 || c >= C) return;
    int n = pout >> 3;             // pout / 8
    feat_out[(long)pout * C + c] = feat_in[(long)n * C + c];
}

// Elementwise add (FP32): out = a + b.
__kernel void add_fp32(
    __global const float* restrict a,
    __global const float* restrict b,
    __global float* restrict out,
    int total)
{
    int i = get_global_id(0);
    if (i >= total) return;
    out[i] = a[i] + b[i];
}
