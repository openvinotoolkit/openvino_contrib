/*
 * Copyright (C) 2018-2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */


// SparseConv3d OpenVINO Extension — Full Implementation
//
// Ports the spconv-based sparse 3D convolution to an OpenVINO custom op
// without any torch/spconv dependency. Uses OpenCL for GPU kernels,
// CPU (with OpenMP) for neighbor map construction.

#include "sparse_conv_3d_op.hpp"
#include <openvino/core/type.hpp>
#include <openvino/core/shape.hpp>

#include <CL/cl.h>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <numeric>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace SAM3DExtension {

// ============================================================
// OpenCL State (lazy-initialized, thread-safe)
// ============================================================
static std::once_flag g_init_flag;
static cl_context       g_context  = nullptr;
static cl_command_queue  g_queue    = nullptr;
static cl_device_id     g_device   = nullptr;
static cl_program       g_program  = nullptr;

// Kernels
// SLM (Shared Local Memory) kernels — significant speedup for large C_in
// Ported from BEVFusion sparse_encoder with tiled extension for C_in > 1213.
static cl_kernel g_subm_conv_fp16_slm_v2_kernel    = nullptr;  // 4 outputs/WI, C_in <= 1213
static cl_kernel g_subm_conv_fp16_slm_v2_w8_kernel = nullptr;  // 8 outputs/WI, C_in <= 1213
static cl_kernel g_subm_conv_fp16_slm_tiled_kernel = nullptr;  // tiled, C_in <= 2048
static cl_kernel g_subm_conv_fp16_slm_k1_kernel    = nullptr;  // per-neighbor SLM, FP32 acc
static cl_kernel g_subm_conv_fp16_slm_k1_f16_kernel= nullptr;  // per-neighbor SLM + FP16 acc + 4-unroll
static cl_kernel g_subm_conv_fp16_slm_k1_int8_kernel= nullptr; // per-neighbor SLM + INT8 weights
static cl_kernel g_subm_conv_fp16_slm_mv_kernel    = nullptr;  // multi-voxel SLM, NV=4 weight sharing
static cl_kernel g_subm_conv_fp16_slm_vtile_kernel = nullptr;  // in-register weight reuse, VT=4 voxels/thread
static cl_kernel g_strided_conv_fp16_kernel        = nullptr;
static cl_kernel g_strided_conv_gather_fp16_kernel = nullptr;
static cl_kernel g_inverse_conv_fp16_kernel        = nullptr;
static cl_kernel g_residual_add_relu_fp16_kernel   = nullptr;
static cl_kernel g_apply_bn_relu_kernel            = nullptr;
static cl_kernel g_float_to_half_kernel            = nullptr;
static cl_kernel g_linear_fp16_kernel              = nullptr;
static cl_kernel g_downsample_avg_kernel           = nullptr;
static cl_kernel g_upsample_nn_kernel              = nullptr;
static cl_kernel g_sparse_to_dense_fp16_kernel     = nullptr;
// Gather-linear-scatter kernels
static cl_kernel g_gather_fp16_kernel               = nullptr;
static cl_kernel g_scatter_add_fp16_kernel          = nullptr;
static cl_kernel g_half_to_float_kernel             = nullptr;
// GPU LayerNorm + SiLU kernels (fused resblock path)
static cl_kernel g_layernorm_silu_fp16_kernel       = nullptr;
static cl_kernel g_layernorm_ss_silu_fp16_kernel    = nullptr;  // scale/shift variant
static cl_kernel g_residual_add_fp16_kernel         = nullptr;
static cl_kernel g_copy_fp16_kernel                 = nullptr;
// FP32 mesh upsample kernels (SparseMeshUpsample op)
static cl_kernel g_gn_reduce_fp32_kernel            = nullptr;
static cl_kernel g_gn_reduce_sqdev_fp32_kernel      = nullptr;
static cl_kernel g_gn_partial_fp32_kernel           = nullptr;
static cl_kernel g_gn_partial_sqdev_fp32_kernel     = nullptr;
static cl_kernel g_gn_affine_silu_fp32_kernel       = nullptr;
static cl_kernel g_subm_conv3d_fp32_kernel          = nullptr;
static cl_kernel g_subm_conv3d_fp32_tiled_kernel    = nullptr;
static cl_kernel g_linear_fp32_kernel               = nullptr;
static cl_kernel g_subdivide_tile_fp32_kernel       = nullptr;
static cl_kernel g_add_fp32_kernel                  = nullptr;

// Persistent GPU buffers
constexpr int MAX_BUFS = 4;
static cl_mem g_feat_buf[MAX_BUFS]     = {};
static size_t g_feat_buf_sz[MAX_BUFS]  = {};
static cl_mem g_identity_buf           = nullptr;
static size_t g_identity_buf_sz        = 0;
static cl_mem g_emb_scale_buf          = nullptr;
static size_t g_emb_scale_buf_sz       = 0;
static cl_mem g_emb_shift_buf          = nullptr;
static size_t g_emb_shift_buf_sz       = 0;
static cl_mem g_neighbor_map_buf       = nullptr;
static cl_mem g_coords_buf            = nullptr;
static cl_mem g_weights_buf           = nullptr;
static cl_mem g_scale_buf             = nullptr;
static cl_mem g_bias_buf              = nullptr;
static cl_mem g_temp_fp32_buf         = nullptr;
static cl_mem g_hash_keys_buf         = nullptr;
static cl_mem g_hash_vals_buf         = nullptr;
static cl_mem g_strided_hash_buf      = nullptr;
static cl_mem g_strided_feat_buf      = nullptr;
static cl_mem g_output_coords_buf     = nullptr;
static size_t g_idb_sz = 0, g_nmb_sz = 0, g_cdb_sz = 0;
static size_t g_wb_sz = 0, g_sb_sz = 0, g_bb_sz = 0, g_t32_sz = 0;
static size_t g_hk_sz = 0, g_hv_sz = 0, g_shb_sz = 0, g_sfb_sz = 0;
static size_t g_ocb_sz = 0;

// Per-layer weight buffers (pre-uploaded FP16)
constexpr int MAX_LAYERS = 64;
static cl_mem g_layer_weights[MAX_LAYERS] = {};
static cl_mem g_layer_scales[MAX_LAYERS]  = {};
static cl_mem g_layer_biases[MAX_LAYERS]  = {};
static bool   g_weights_uploaded = false;


// ============================================================
// Neighbor Map Cache — avoid rebuilding when coordinates unchanged
// ============================================================
// In SLAT ODE, the same voxel coordinates are used for all 25 steps
// (352+ conv calls). Caching the neighbor map saves ~95% of rebuilds.
static std::vector<int32_t> g_cached_coords;     // last coordinates used
static std::vector<int32_t> g_cached_nmap;        // cached neighbor map
static int                  g_cached_nmap_N = 0;  // N voxels for cached map

// ============================================================
// Async Feature Prefetch State
// ============================================================
// Python can call sam3d_prefetch_features() to start a non-blocking DMA
// of feature data to g_feat_fp32_buf. When gpu_fused_resblock() is called
// with the same pointer+size, it skips the blocking map+memcpy and just
// waits for the in-flight event.
static cl_command_queue g_prefetch_queue = nullptr;  // separate queue for async DMA
static cl_event         g_prefetch_event  = nullptr;  // event for in-flight DMA
static const void*      g_prefetch_ptr    = nullptr;  // source pointer used in prefetch
static size_t           g_prefetch_sz     = 0;        // size of in-flight DMA

// GPU-side neighbor-map cache state (for gpu_fused_resblock). Tracks whether
// g_neighbor_map_buf currently holds the nmap for N voxels. Invalidated whenever
// any other path overwrites g_neighbor_map_buf.
static bool                 g_gpu_nmap_valid = false;
static int                  g_gpu_nmap_N     = 0;

static bool coords_match(const int32_t* coords, int N) {
    if (N != g_cached_nmap_N) return false;
    size_t sz = (size_t)N * 4;
    return std::memcmp(coords, g_cached_coords.data(), sz * sizeof(int32_t)) == 0;
}

// ============================================================
// Weight Cache — convert FP32→FP16 and upload to GPU once per unique weight set
// ============================================================
// Key: pointer to weight data (stable across calls for same OV compiled model)
// Value: pre-uploaded GPU cl_mem buffers
struct CachedWeights {
    cl_mem weights_fp16 = nullptr;   // [nk * cin * cout] FP16
    cl_mem scale        = nullptr;   // [cout] FP32
    cl_mem bias         = nullptr;   // [cout] FP32
    size_t w_bytes = 0, s_bytes = 0;
    // INT8 weight quantization (optional — set when INT8 weights are available)
    cl_mem weights_int8 = nullptr;   // [nk * cin * cout] INT8 (1 byte each)
    cl_mem w_quant_scale= nullptr;   // [nk * cout] FP16 — per-kernel per-out-ch scale
    size_t w_int8_bytes = 0;
    bool   has_int8     = false;
};
static std::unordered_map<const float*, CachedWeights> g_weight_cache;

// ============================================================
// Pre-allocated CPU buffers — avoid per-call heap allocations
// ============================================================
static std::vector<float>    g_static_buf_a;
static std::vector<float>    g_static_buf_b;
static std::vector<float>    g_static_buf_identity;
// GPU buffer for FP32 features (for GPU-side FP32→FP16 conversion)
static cl_mem  g_feat_fp32_buf     = nullptr;
static size_t  g_feat_fp32_buf_sz  = 0;

#define CL_CHECK(call) do { \
    cl_int err_ = (call); \
    if (err_ != CL_SUCCESS) \
        throw std::runtime_error("OpenCL error " + std::to_string(err_) + \
            " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__)); \
} while(0)

static void ensure_buf(cl_mem& buf, size_t& cur, size_t need, cl_mem_flags flags) {
    if (need > cur) {
        if (buf) clReleaseMemObject(buf);
        cl_int e;
        buf = clCreateBuffer(g_context, flags, need, nullptr, &e);
        CL_CHECK(e);
        cur = need;
    }
}

static uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint16_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C00;
    return sign | (uint16_t)(exp << 10) | (uint16_t)(mant >> 13);
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) { // subnormal
        if (mant == 0) { float r; uint32_t v = sign; std::memcpy(&r, &v, 4); return r; }
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        exp++; mant &= ~0x400;
    } else if (exp == 31) {
        exp = 255;
    }
    uint32_t v = sign | ((exp + 112) << 23) | (mant << 13);
    float r;
    std::memcpy(&r, &v, 4);
    return r;
}

static std::vector<uint16_t> to_fp16(const float* d, int n) {
    std::vector<uint16_t> out(n);
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < n; i++) out[i] = f32_to_f16(d[i]);
    return out;
}

static void upload(cl_mem buf, const void* data, size_t sz) {
    CL_CHECK(clEnqueueWriteBuffer(g_queue, buf, CL_FALSE, 0, sz, data, 0, nullptr, nullptr));
}

static void download(cl_mem buf, void* data, size_t sz) {
    CL_CHECK(clEnqueueReadBuffer(g_queue, buf, CL_TRUE, 0, sz, data, 0, nullptr, nullptr));
}

// ============================================================
// CPU Hash Table for Neighbor Map Construction
// ============================================================
struct HashTable {
    static constexpr int EMPTY = -1;
    std::vector<int64_t> keys;
    std::vector<int32_t> vals;
    int capacity;

    explicit HashTable(int cap) : capacity(cap), keys(cap, INT64_MAX), vals(cap, EMPTY) {}

    inline int hash(int64_t key) const {
        // FNV-1a inspired hash
        uint64_t h = (uint64_t)key;
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;
        return (int)(h & (uint64_t)(capacity - 1));
    }

    void insert(int64_t key, int32_t val) {
        int h = hash(key);
        while (true) {
            if (keys[h] == INT64_MAX) {
                keys[h] = key;
                vals[h] = val;
                return;
            }
            if (keys[h] == key) {
                vals[h] = val;  // overwrite
                return;
            }
            h = (h + 1) & (capacity - 1);
        }
    }

    int32_t find(int64_t key) const {
        int h = hash(key);
        while (true) {
            if (keys[h] == INT64_MAX) return EMPTY;
            if (keys[h] == key) return vals[h];
            h = (h + 1) & (capacity - 1);
        }
    }
};

static inline int64_t coord_key(int x, int y, int z) {
    return ((int64_t)x << 20) | ((int64_t)y << 10) | (int64_t)z;
}

// Build neighbor map for SubMConv3d (3×3×3 kernel)
// neighbor_map: [N, 27] — for each voxel, index of each of 27 neighbors (-1 if absent)
static void build_subm_neighbor_map(
    const int32_t* coords,  // [N, 4] (batch, x, y, z)
    int N,
    int32_t* neighbor_map   // output [N, 27]
) {
    // Build hash table
    int ht_cap = 1;
    while (ht_cap < N * 4) ht_cap <<= 1;
    if (ht_cap < 1024) ht_cap = 1024;
    HashTable ht(ht_cap);

    for (int i = 0; i < N; i++) {
        int bx = coords[i * 4 + 0];
        int x  = coords[i * 4 + 1];
        int y  = coords[i * 4 + 2];
        int z  = coords[i * 4 + 3];
        ht.insert(coord_key(x + bx * 1024, y, z), i);
    }

    // Build neighbor map
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 256)
    #endif
    for (int i = 0; i < N; i++) {
        int bx = coords[i * 4 + 0];
        int x  = coords[i * 4 + 1];
        int y  = coords[i * 4 + 2];
        int z  = coords[i * 4 + 3];

        int idx = 0;
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dz = -1; dz <= 1; dz++) {
                    int nx = x + dx, ny = y + dy, nz = z + dz;
                    int64_t key = coord_key(nx + bx * 1024, ny, nz);
                    neighbor_map[i * 27 + idx] = ht.find(key);
                    idx++;
                }
            }
        }
    }
}

// Build strided conv output coords and mapping
// Returns number of output voxels
static int build_strided_conv_map(
    const int32_t* coords_in,  // [N_in, 4]
    int N_in,
    int stride,
    int32_t* coords_out,       // [MAX_N, 4] output coords
    int32_t* in_to_out_map,    // [N_in] — maps each input voxel to output voxel index
    int max_out
) {
    // Hash table for output coordinates
    int ht_cap = 1;
    while (ht_cap < N_in * 2) ht_cap <<= 1;
    if (ht_cap < 1024) ht_cap = 1024;
    HashTable ht(ht_cap);

    int N_out = 0;
    for (int i = 0; i < N_in; i++) {
        int b  = coords_in[i * 4 + 0];
        int ox = coords_in[i * 4 + 1] / stride;
        int oy = coords_in[i * 4 + 2] / stride;
        int oz = coords_in[i * 4 + 3] / stride;
        int64_t key = coord_key(ox + b * 1024, oy, oz);
        int32_t existing = ht.find(key);
        if (existing == HashTable::EMPTY) {
            if (N_out >= max_out) {
                in_to_out_map[i] = -1;
                continue;
            }
            ht.insert(key, N_out);
            coords_out[N_out * 4 + 0] = b;
            coords_out[N_out * 4 + 1] = ox;
            coords_out[N_out * 4 + 2] = oy;
            coords_out[N_out * 4 + 3] = oz;
            in_to_out_map[i] = N_out;
            N_out++;
        } else {
            in_to_out_map[i] = existing;
        }
    }
    return N_out;
}

// ============================================================
// CPU Reference Implementation (SubMConv3d)
// Vectorized: iterate over kernel positions, gather-GEMM-scatter
// ============================================================
static void cpu_subm_conv3d(
    const float* in_feats,     // [N, C_in]
    const float* weights,      // [27, C_in, C_out] or flat [27*C_in*C_out]
    const float* scale,        // [C_out] — fused BN scale (gamma/sqrt(var+eps))
    const float* bias,         // [C_out] — fused BN bias
    const int32_t* nmap,       // [N, 27]
    int N, int C_in, int C_out,
    float* out_feats           // [N, C_out]
) {
    // Zero output
    std::memset(out_feats, 0, (size_t)N * C_out * sizeof(float));

    // For each of 27 kernel positions, collect all valid (i→j) pairs,
    // then compute a batched GEMM: out[i] += in[j] * W[k]
    // This is cache-friendly and easily vectorized by the compiler.
    std::vector<int32_t> src_idx, dst_idx;
    src_idx.reserve(N);
    dst_idx.reserve(N);

    // Temporary accumulation buffer for gathered inputs
    std::vector<float> gathered(N * C_in);

    for (int k = 0; k < 27; k++) {
        const float* W_k = weights + (size_t)k * C_in * C_out;

        // Collect valid pairs for this kernel position
        src_idx.clear();
        dst_idx.clear();
        for (int i = 0; i < N; i++) {
            int j = nmap[(size_t)i * 27 + k];
            if (j >= 0) {
                src_idx.push_back(j);  // input index
                dst_idx.push_back(i);  // output index
            }
        }

        int M = (int)src_idx.size();
        if (M == 0) continue;

        // Gather: gathered[m, :] = in_feats[src_idx[m], :]
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int m = 0; m < M; m++) {
            std::memcpy(&gathered[(size_t)m * C_in],
                        &in_feats[(size_t)src_idx[m] * C_in],
                        C_in * sizeof(float));
        }

        // GEMM: tmp[m, co] = sum_ci gathered[m, ci] * W_k[ci, co]
        // scattered directly into out_feats[dst_idx[m], co]
        // Note: weights layout is [C_in, C_out] (row-major, C_out is inner dim)
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int m = 0; m < M; m++) {
            const float* g_row = &gathered[(size_t)m * C_in];
            float* o_row = &out_feats[(size_t)dst_idx[m] * C_out];
            for (int ci = 0; ci < C_in; ci++) {
                float g_val = g_row[ci];
                const float* w_row = W_k + (size_t)ci * C_out;
                for (int co = 0; co < C_out; co++) {
                    o_row[co] += g_val * w_row[co];
                }
            }
        }
    }

    // Apply fused BN: out[i][c] = out[i][c] * scale[c] + bias[c]
    if (scale && bias) {
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int i = 0; i < N; i++) {
            float* o_i = out_feats + (size_t)i * C_out;
            for (int c = 0; c < C_out; c++) {
                o_i[c] = o_i[c] * scale[c] + bias[c];
            }
        }
    }
}

// CPU linear (per-voxel)
static void cpu_linear(
    const float* in_feats,   // [N, C_in]
    const float* weights,    // [C_in, C_out]
    const float* bias,       // [C_out] (may be null)
    int N, int C_in, int C_out,
    float* out_feats         // [N, C_out]
) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 64)
    #endif
    for (int i = 0; i < N; i++) {
        const float* f_i = in_feats + (size_t)i * C_in;
        float* o_i = out_feats + (size_t)i * C_out;
        for (int co = 0; co < C_out; co++) {
            float acc = bias ? bias[co] : 0.0f;
            for (int ci = 0; ci < C_in; ci++) {
                acc += f_i[ci] * weights[ci * C_out + co];
            }
            o_i[co] = acc;
        }
    }
}

// CPU ReLU in-place
static void cpu_relu(float* data, size_t n) {
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < n; i++) {
        if (data[i] < 0.0f) data[i] = 0.0f;
    }
}

// CPU SiLU in-place
static void cpu_silu(float* data, size_t n) {
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < n; i++) {
        data[i] = data[i] / (1.0f + std::exp(-data[i]));
    }
}

// CPU residual add: out = a + b
static void cpu_residual_add(const float* a, const float* b, float* out, size_t n) {
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

// CPU LayerNorm (per-voxel, across channels)
static void cpu_layernorm(
    const float* in_feats,   // [N, C]
    float* out_feats,        // [N, C]
    const float* gamma,      // [C] (may be null for no affine)
    const float* beta,       // [C] (may be null)
    int N, int C,
    float eps = 1e-6f
) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 64)
    #endif
    for (int i = 0; i < N; i++) {
        const float* fi = in_feats + (size_t)i * C;
        float* fo = out_feats + (size_t)i * C;
        // Compute mean
        float mean = 0.0f;
        for (int c = 0; c < C; c++) mean += fi[c];
        mean /= (float)C;
        // Compute variance
        float var = 0.0f;
        for (int c = 0; c < C; c++) {
            float d = fi[c] - mean;
            var += d * d;
        }
        var /= (float)C;
        float inv_std = 1.0f / std::sqrt(var + eps);
        // Normalize
        for (int c = 0; c < C; c++) {
            float normalized = (fi[c] - mean) * inv_std;
            if (gamma) normalized = normalized * gamma[c];
            if (beta) normalized = normalized + beta[c];
            fo[c] = normalized;
        }
    }
}

// CPU downsample (coordinate-based average pooling)
static int cpu_downsample(
    const float* in_feats,     // [N_in, C]
    const int32_t* in_coords,  // [N_in, 4]
    int N_in, int C, int factor,
    float* out_feats,          // [MAX_N, C] (will be filled)
    int32_t* out_coords,       // [MAX_N, 4]
    int32_t* in_to_out,        // [N_in] — mapping from input to output voxel
    int max_out
) {
    // Build output coordinate set
    int N_out = build_strided_conv_map(in_coords, N_in, factor,
                                       out_coords, in_to_out, max_out);

    // Accumulate features + count
    std::vector<int> counts(N_out, 0);
    std::memset(out_feats, 0, (size_t)N_out * C * sizeof(float));

    for (int i = 0; i < N_in; i++) {
        int out_idx = in_to_out[i];
        if (out_idx < 0) continue;
        float* dst = out_feats + (size_t)out_idx * C;
        const float* src = in_feats + (size_t)i * C;
        for (int c = 0; c < C; c++) {
            dst[c] += src[c];
        }
        counts[out_idx]++;
    }

    // Average
    for (int j = 0; j < N_out; j++) {
        if (counts[j] > 1) {
            float inv = 1.0f / (float)counts[j];
            float* f = out_feats + (size_t)j * C;
            for (int c = 0; c < C; c++) f[c] *= inv;
        }
    }

    return N_out;
}

// CPU upsample (nearest neighbor via cached mapping)
static void cpu_upsample(
    const float* in_feats,     // [N_in, C]
    const int32_t* upsample_idx, // [N_out] — maps each output voxel to source input voxel
    int N_out, int C,
    float* out_feats           // [N_out, C]
) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 64)
    #endif
    for (int j = 0; j < N_out; j++) {
        int src = upsample_idx[j];
        if (src < 0) {
            std::memset(out_feats + (size_t)j * C, 0, C * sizeof(float));
        } else {
            std::memcpy(out_feats + (size_t)j * C,
                       in_feats + (size_t)src * C,
                       C * sizeof(float));
        }
    }
}

// ============================================================
// Context Injection API
// ============================================================
// When the host injects an existing cl_context (e.g. from the OV GPU plugin),
// init_opencl() will skip creating its own context and use the injected one.
// This allows the extension and OV GPU plugin to share one cl_context,
// eliminating host round-trips at every extension↔OV boundary.

static uint64_t g_injected_cl_context = 0;
static uint64_t g_injected_cl_queue   = 0;

extern "C" {
    // Inject an existing cl_context + cl_command_queue before first use.
    // Pass 0 for cl_queue to let the extension create its own queue on the context.
    void sam3d_inject_cl_context(uint64_t ctx_handle, uint64_t queue_handle) {
        g_injected_cl_context = ctx_handle;
        g_injected_cl_queue   = queue_handle;
    }

    // Returns the cl_context used by the extension (as uint64_t).
    // If not yet initialized, returns 0.
    uint64_t sam3d_get_cl_context() {
        return (uint64_t)(uintptr_t)g_context;
    }

    // Returns the cl_command_queue used by the extension (as uint64_t).
    uint64_t sam3d_get_cl_queue() {
        return (uint64_t)(uintptr_t)g_queue;
    }

    // Start an async DMA prefetch of feature data to g_feat_fp32_buf.
    // Must be called AFTER init_opencl() (i.e., after the first extension op).
    // feat_ptr: host pointer to FP32 features [N, C_in]
    // n_bytes: sizeof(float) * N * C_in
    // Returns 1 on success, 0 if not ready (not yet initialized).
    int sam3d_prefetch_features(const void* feat_ptr, size_t n_bytes) {
        if (!g_context || !g_feat_fp32_buf || g_feat_fp32_buf_sz < n_bytes) return 0;
        // Cancel any previous in-flight event
        if (g_prefetch_event) {
            clWaitForEvents(1, &g_prefetch_event);
            clReleaseEvent(g_prefetch_event);
            g_prefetch_event = nullptr;
        }
        // Create prefetch queue if needed
        if (!g_prefetch_queue) {
            cl_int e;
            g_prefetch_queue = clCreateCommandQueue(g_context, g_device, 0, &e);
            if (e != CL_SUCCESS) return 0;
        }
        // Async write: DMA from host to g_feat_fp32_buf without blocking
        cl_event ev;
        cl_int e = clEnqueueWriteBuffer(g_prefetch_queue, g_feat_fp32_buf, CL_FALSE,
                                         0, n_bytes, feat_ptr, 0, nullptr, &ev);
        if (e != CL_SUCCESS) return 0;
        clFlush(g_prefetch_queue);
        g_prefetch_event = ev;
        g_prefetch_ptr   = feat_ptr;
        g_prefetch_sz    = n_bytes;
        return 1;
    }
} // extern "C"

// ============================================================
// OpenCL Initialization
// ============================================================
static void init_opencl() {
    cl_int err;

    // ── Injected context path ──
    // If a cl_context was injected (e.g. from OV GPU plugin), use it directly
    // and skip platform/device enumeration entirely.
    if (g_injected_cl_context != 0) {
        g_context = (cl_context)(uintptr_t)g_injected_cl_context;
        // Retain the context so clReleaseContext at shutdown doesn't double-free
        clRetainContext(g_context);
        // Retrieve the device from the injected context
        size_t dev_size = 0;
        clGetContextInfo(g_context, CL_CONTEXT_DEVICES, 0, nullptr, &dev_size);
        if (dev_size >= sizeof(cl_device_id)) {
            clGetContextInfo(g_context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &g_device, nullptr);
        }
        if (g_injected_cl_queue != 0) {
            g_queue = (cl_command_queue)(uintptr_t)g_injected_cl_queue;
            clRetainCommandQueue(g_queue);
        } else {
            // Create our own in-order queue on the injected context+device
            g_queue = clCreateCommandQueue(g_context, g_device, 0, &err);
            CL_CHECK(err);
        }
        std::cout << "[sparse_conv_3d] Using injected OpenCL context " << (void*)g_context
                  << " queue " << (void*)g_queue << std::endl;
    } else {
    // ── Normal self-init path ──
    cl_uint npf;
    err = clGetPlatformIDs(0, nullptr, &npf);
    if (err != CL_SUCCESS || npf == 0)
        throw std::runtime_error("No OpenCL platforms");

    std::vector<cl_platform_id> plats(npf);
    clGetPlatformIDs(npf, plats.data(), nullptr);

    // Try Intel GPU first, then any GPU
    bool found = false;
    for (int pass = 0; pass < 2 && !found; pass++) {
        for (auto& p : plats) {
            cl_uint nd;
            if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) != CL_SUCCESS || nd == 0)
                continue;
            std::vector<cl_device_id> devs(nd);
            clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, devs.data(), nullptr);
            for (auto& d : devs) {
                char vendor[256];
                clGetDeviceInfo(d, CL_DEVICE_VENDOR, sizeof(vendor), vendor, nullptr);
                bool is_intel = (std::string(vendor).find("Intel") != std::string::npos);
                if (pass == 0 && !is_intel) continue;  // first pass: Intel only

                g_device = d;
                g_context = clCreateContext(nullptr, 1, &g_device, nullptr, nullptr, &err);
                CL_CHECK(err);
                g_queue = clCreateCommandQueue(g_context, g_device, 0, &err);
                CL_CHECK(err);

                // Load OpenCL kernel source
                std::string this_dir = std::string(__FILE__);
                {
                    auto pos = this_dir.find_last_of("/\\");
                    if (pos != std::string::npos) this_dir = this_dir.substr(0, pos);
                    else this_dir = ".";
                }
                std::string src;
                for (const auto& path : {
                    this_dir + "/sparse_conv_3d.cl",
                }) {
                    std::ifstream f(path);
                    if (f.is_open()) {
                        std::ostringstream ss;
                        ss << f.rdbuf();
                        src = ss.str();
                        break;
                    }
                }
                if (src.empty())
                    throw std::runtime_error("Cannot find sparse_conv_3d.cl");

                const char* sp = src.c_str();
                size_t sl = src.size();

                // Try to load pre-compiled binary (fast path)
                const std::string cache_path = "/tmp/sam3d_sparse_conv_3d_kernel.bin";
                // Compute a simple hash of the source so the cache is invalidated
                // automatically whenever the .cl file changes.
                uint32_t src_hash = 2166136261u;
                for (char c : src) {
                    src_hash ^= (uint8_t)c;
                    src_hash *= 16777619u;
                }
                const std::string hash_path = cache_path + ".hash";
                bool loaded_from_cache = false;
                // Check if cached hash matches current source
                {
                    std::ifstream hf(hash_path);
                    uint32_t cached_hash = 0;
                    if (hf >> cached_hash && cached_hash == src_hash) {
                        std::ifstream cf(cache_path, std::ios::binary | std::ios::ate);
                        if (cf.is_open()) {
                            size_t bin_sz = cf.tellg();
                            cf.seekg(0);
                            std::vector<uint8_t> bin(bin_sz);
                            cf.read(reinterpret_cast<char*>(bin.data()), bin_sz);
                            if (cf.good()) {
                                const uint8_t* bp2 = bin.data();
                                cl_int bin_status;
                                cl_program p2 = clCreateProgramWithBinary(
                                    g_context, 1, &g_device, &bin_sz, &bp2, &bin_status, &err);
                                if (err == CL_SUCCESS && bin_status == CL_SUCCESS) {
                                    cl_build_status bstatus;
                                    err = clGetProgramBuildInfo(p2, g_device, CL_PROGRAM_BUILD_STATUS,
                                        sizeof(bstatus), &bstatus, nullptr);
                                    if (err == CL_SUCCESS && bstatus == CL_BUILD_SUCCESS) {
                                        cl_int ke;
                                        cl_kernel test_k = clCreateKernel(p2, "subm_conv_fp16", &ke);
                                        if (ke == CL_SUCCESS) {
                                            clReleaseKernel(test_k);
                                            g_program = p2;
                                            loaded_from_cache = true;
                                        } else {
                                            err = clBuildProgram(p2, 1, &g_device, nullptr, nullptr, nullptr);
                                            if (err == CL_SUCCESS) {
                                                g_program = p2;
                                                loaded_from_cache = true;
                                            } else {
                                                clReleaseProgram(p2);
                                            }
                                        }
                                    } else {
                                        err = clBuildProgram(p2, 1, &g_device, nullptr, nullptr, nullptr);
                                        if (err == CL_SUCCESS) {
                                            g_program = p2;
                                            loaded_from_cache = true;
                                        } else {
                                            clReleaseProgram(p2);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }  // end hash+binary cache block

                if (!loaded_from_cache) {
                    // Compile from source
                    g_program = clCreateProgramWithSource(g_context, 1, &sp, &sl, &err);
                    CL_CHECK(err);

                    err = clBuildProgram(g_program, 1, &g_device,
                        "-cl-std=CL2.0 -cl-mad-enable",
                        nullptr, nullptr);
                    if (err != CL_SUCCESS) {
                        char log[16384];
                        size_t ll;
                        clGetProgramBuildInfo(g_program, g_device, CL_PROGRAM_BUILD_LOG,
                                             sizeof(log), log, &ll);
                        throw std::runtime_error(std::string("Kernel build failed:\n") + log);
                    }

                    // Save binary + hash for future fast loading
                    size_t bins_sz = 0;
                    clGetProgramInfo(g_program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bins_sz, nullptr);
                    if (bins_sz > 0) {
                        std::vector<uint8_t> bin(bins_sz);
                        uint8_t* bp2 = bin.data();
                        clGetProgramInfo(g_program, CL_PROGRAM_BINARIES, sizeof(uint8_t*), &bp2, nullptr);
                        std::ofstream of(cache_path, std::ios::binary);
                        of.write(reinterpret_cast<const char*>(bin.data()), bins_sz);
                        // Write hash so stale binaries are detected next run
                        std::ofstream hfw(hash_path);
                        hfw << src_hash;
                    }
                }

                // Create kernels
                g_subm_conv_fp16_slm_v2_kernel = clCreateKernel(g_program, "subm_conv_fp16_slm_v2", &err);
                if (err != CL_SUCCESS) {
                    std::cout << "[sparse_conv_3d] subm_conv_fp16_slm_v2 not found" << std::endl;
                    g_subm_conv_fp16_slm_v2_kernel = nullptr;
                }
                g_subm_conv_fp16_slm_v2_w8_kernel = clCreateKernel(g_program, "subm_conv_fp16_slm_v2_w8", &err);
                if (err != CL_SUCCESS) {
                    std::cout << "[sparse_conv_3d] subm_conv_fp16_slm_v2_w8 not found" << std::endl;
                    g_subm_conv_fp16_slm_v2_w8_kernel = nullptr;
                }
                g_subm_conv_fp16_slm_tiled_kernel = clCreateKernel(g_program, "subm_conv_fp16_slm_tiled", &err);
                if (err != CL_SUCCESS) {
                    std::cout << "[sparse_conv_3d] subm_conv_fp16_slm_tiled not found" << std::endl;
                    g_subm_conv_fp16_slm_tiled_kernel = nullptr;
                }
                g_subm_conv_fp16_slm_k1_kernel = clCreateKernel(g_program, "subm_conv_fp16_slm_k1", &err);
                if (err != CL_SUCCESS) {
                    std::cout << "[sparse_conv_3d] subm_conv_fp16_slm_k1 not found" << std::endl;
                    g_subm_conv_fp16_slm_k1_kernel = nullptr;
                }
                g_subm_conv_fp16_slm_k1_f16_kernel = clCreateKernel(g_program, "subm_conv_fp16_slm_k1_f16", &err);
                if (err != CL_SUCCESS) {
                    std::cout << "[sparse_conv_3d] subm_conv_fp16_slm_k1_f16 not found" << std::endl;
                    g_subm_conv_fp16_slm_k1_f16_kernel = nullptr;
                }
                g_subm_conv_fp16_slm_k1_int8_kernel = clCreateKernel(g_program, "subm_conv_fp16_slm_k1_int8", &err);
                if (err != CL_SUCCESS) {
                    std::cout << "[sparse_conv_3d] subm_conv_fp16_slm_k1_int8 not found" << std::endl;
                    g_subm_conv_fp16_slm_k1_int8_kernel = nullptr;
                }
                g_subm_conv_fp16_slm_mv_kernel = clCreateKernel(g_program, "subm_conv_fp16_slm_mv", &err);
                if (err != CL_SUCCESS) {
                    std::cout << "[sparse_conv_3d] subm_conv_fp16_slm_mv not found" << std::endl;
                    g_subm_conv_fp16_slm_mv_kernel = nullptr;
                }
                g_subm_conv_fp16_slm_vtile_kernel = clCreateKernel(g_program, "subm_conv_fp16_slm_vtile", &err);
                if (err != CL_SUCCESS) {
                    std::cout << "[sparse_conv_3d] subm_conv_fp16_slm_vtile not found" << std::endl;
                    g_subm_conv_fp16_slm_vtile_kernel = nullptr;
                }

                g_strided_conv_fp16_kernel = clCreateKernel(g_program, "strided_conv_scatter_fp16", &err); CL_CHECK(err);
                g_strided_conv_gather_fp16_kernel = clCreateKernel(g_program, "strided_conv_gather_fp16", &err); CL_CHECK(err);
                g_inverse_conv_fp16_kernel = clCreateKernel(g_program, "inverse_conv_fp16", &err); CL_CHECK(err);
                g_residual_add_relu_fp16_kernel = clCreateKernel(g_program, "residual_add_relu_fp16", &err); CL_CHECK(err);
                g_apply_bn_relu_kernel = clCreateKernel(g_program, "apply_bn_relu", &err); CL_CHECK(err);
                g_float_to_half_kernel = clCreateKernel(g_program, "float_to_half_kernel", &err); CL_CHECK(err);
                g_linear_fp16_kernel = clCreateKernel(g_program, "linear_fp16", &err); CL_CHECK(err);
                g_downsample_avg_kernel = clCreateKernel(g_program, "downsample_avg_fp16", &err); CL_CHECK(err);
                g_upsample_nn_kernel = clCreateKernel(g_program, "upsample_nn_fp16", &err); CL_CHECK(err);
                g_sparse_to_dense_fp16_kernel = clCreateKernel(g_program, "sparse_to_dense_fp16", &err); CL_CHECK(err);
                // Gather-linear-scatter kernels
                g_gather_fp16_kernel = clCreateKernel(g_program, "gather_fp16", &err); CL_CHECK(err);
                g_scatter_add_fp16_kernel = clCreateKernel(g_program, "scatter_add_fp16", &err); CL_CHECK(err);
                g_half_to_float_kernel = clCreateKernel(g_program, "half_to_float_kernel", &err); CL_CHECK(err);
                g_layernorm_silu_fp16_kernel = clCreateKernel(g_program, "layernorm_silu_fp16", &err); CL_CHECK(err);
                g_layernorm_ss_silu_fp16_kernel = clCreateKernel(g_program, "layernorm_scale_shift_silu_fp16", &err); CL_CHECK(err);
                g_residual_add_fp16_kernel = clCreateKernel(g_program, "residual_add_fp16", &err); CL_CHECK(err);
                g_copy_fp16_kernel = clCreateKernel(g_program, "copy_fp16", &err); CL_CHECK(err);

                // FP32 mesh upsample kernels
                g_gn_reduce_fp32_kernel = clCreateKernel(g_program, "gn_reduce_fp32", &err); CL_CHECK(err);
                g_gn_reduce_sqdev_fp32_kernel = clCreateKernel(g_program, "gn_reduce_sqdev_fp32", &err); CL_CHECK(err);
                g_gn_partial_fp32_kernel = clCreateKernel(g_program, "gn_partial_fp32", &err); CL_CHECK(err);
                g_gn_partial_sqdev_fp32_kernel = clCreateKernel(g_program, "gn_partial_sqdev_fp32", &err); CL_CHECK(err);
                g_gn_affine_silu_fp32_kernel = clCreateKernel(g_program, "gn_affine_silu_fp32", &err); CL_CHECK(err);
                g_subm_conv3d_fp32_kernel = clCreateKernel(g_program, "subm_conv3d_fp32", &err); CL_CHECK(err);
                g_subm_conv3d_fp32_tiled_kernel = clCreateKernel(g_program, "subm_conv3d_fp32_tiled", &err); CL_CHECK(err);
                g_linear_fp32_kernel = clCreateKernel(g_program, "dense_linear_fp32", &err); CL_CHECK(err);
                g_subdivide_tile_fp32_kernel = clCreateKernel(g_program, "subdivide_tile_fp32", &err); CL_CHECK(err);
                g_add_fp32_kernel = clCreateKernel(g_program, "add_fp32", &err); CL_CHECK(err);

                char dev_name[256];
                clGetDeviceInfo(g_device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr);
                std::cout << "[sparse_conv_3d] Initialized OpenCL on: " << dev_name
                          << " (vendor: " << vendor << ")" << std::endl;

                found = true;
                break;
            }
            if (found) break;
        }
    }
    if (!found)
        throw std::runtime_error("[sparse_conv_3d] No suitable GPU device found for OpenCL");
    } // end else (normal self-init path)

    // ── Common: compile kernels from source or binary cache ──
    // After either the injected-context path or the self-init path, g_context,
    // g_device, and g_queue are set.  The program/kernel compilation is
    // identical in both cases.  We compile here if the injected path was taken
    // (the self-init path already compiled above inside the loop).
    if (g_injected_cl_context != 0) {
        // Load and compile kernels using the injected context
        std::string this_dir = std::string(__FILE__);
        {
            auto pos = this_dir.find_last_of("/\\");
            if (pos != std::string::npos) this_dir = this_dir.substr(0, pos);
            else this_dir = ".";
        }
        std::string src;
        for (const auto& path : { this_dir + "/sparse_conv_3d.cl" }) {
            std::ifstream f(path);
            if (f.is_open()) {
                std::ostringstream ss;
                ss << f.rdbuf();
                src = ss.str();
                break;
            }
        }
        if (src.empty())
            throw std::runtime_error("Cannot find sparse_conv_3d.cl (injected path)");

        const char* sp = src.c_str();
        size_t sl = src.size();

        uint32_t src_hash = 2166136261u;
        for (char c : src) { src_hash ^= (uint8_t)c; src_hash *= 16777619u; }

        // Try binary cache (different tag from self-init to avoid cross-contamination)
        const std::string cache_path = "/tmp/sam3d_sparse_conv_3d_injected.bin";
        const std::string hash_path  = cache_path + ".hash";
        bool loaded_from_cache = false;
        {
            std::ifstream hf(hash_path);
            uint32_t cached_hash = 0;
            if (hf >> cached_hash && cached_hash == src_hash) {
                std::ifstream cf(cache_path, std::ios::binary | std::ios::ate);
                if (cf.is_open()) {
                    size_t bin_sz = cf.tellg(); cf.seekg(0);
                    std::vector<uint8_t> bin(bin_sz);
                    cf.read(reinterpret_cast<char*>(bin.data()), bin_sz);
                    if (cf.good()) {
                        const uint8_t* bp2 = bin.data();
                        cl_int bin_status;
                        cl_program p2 = clCreateProgramWithBinary(
                            g_context, 1, &g_device, &bin_sz, &bp2, &bin_status, &err);
                        if (err == CL_SUCCESS && bin_status == CL_SUCCESS) {
                            err = clBuildProgram(p2, 1, &g_device, nullptr, nullptr, nullptr);
                            if (err == CL_SUCCESS) { g_program = p2; loaded_from_cache = true; }
                            else clReleaseProgram(p2);
                        }
                    }
                }
            }
        }
        if (!loaded_from_cache) {
            g_program = clCreateProgramWithSource(g_context, 1, &sp, &sl, &err); CL_CHECK(err);
            err = clBuildProgram(g_program, 1, &g_device, "-cl-std=CL2.0 -cl-mad-enable", nullptr, nullptr);
            if (err != CL_SUCCESS) {
                char log[16384]; size_t ll;
                clGetProgramBuildInfo(g_program, g_device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, &ll);
                throw std::runtime_error(std::string("Kernel build failed (injected path):\n") + log);
            }
            size_t bins_sz = 0;
            clGetProgramInfo(g_program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bins_sz, nullptr);
            if (bins_sz > 0) {
                std::vector<uint8_t> bin(bins_sz);
                uint8_t* bp2 = bin.data();
                clGetProgramInfo(g_program, CL_PROGRAM_BINARIES, sizeof(uint8_t*), &bp2, nullptr);
                std::ofstream of(cache_path, std::ios::binary);
                of.write(reinterpret_cast<const char*>(bin.data()), bins_sz);
                std::ofstream hfw(hash_path); hfw << src_hash;
            }
        }
        // Create all kernels (same list as self-init path)
        #define MK(k, name) k = clCreateKernel(g_program, name, &err); CL_CHECK(err)
        #define MK_OPT(k, name) k = clCreateKernel(g_program, name, &err); if (err != CL_SUCCESS) k = nullptr
        MK_OPT(g_subm_conv_fp16_slm_v2_kernel, "subm_conv_fp16_slm_v2");
        MK_OPT(g_subm_conv_fp16_slm_v2_w8_kernel, "subm_conv_fp16_slm_v2_w8");
        MK_OPT(g_subm_conv_fp16_slm_tiled_kernel, "subm_conv_fp16_slm_tiled");
        MK_OPT(g_subm_conv_fp16_slm_k1_kernel, "subm_conv_fp16_slm_k1");
        MK_OPT(g_subm_conv_fp16_slm_k1_f16_kernel, "subm_conv_fp16_slm_k1_f16");
        MK_OPT(g_subm_conv_fp16_slm_k1_int8_kernel, "subm_conv_fp16_slm_k1_int8");
        MK_OPT(g_subm_conv_fp16_slm_mv_kernel, "subm_conv_fp16_slm_mv");
        MK_OPT(g_subm_conv_fp16_slm_vtile_kernel, "subm_conv_fp16_slm_vtile");
        MK(g_strided_conv_fp16_kernel,         "strided_conv_scatter_fp16");
        MK(g_strided_conv_gather_fp16_kernel,  "strided_conv_gather_fp16");
        MK(g_inverse_conv_fp16_kernel,         "inverse_conv_fp16");
        MK(g_residual_add_relu_fp16_kernel,    "residual_add_relu_fp16");
        MK(g_apply_bn_relu_kernel,             "apply_bn_relu");
        MK(g_float_to_half_kernel,             "float_to_half_kernel");
        MK(g_linear_fp16_kernel,               "linear_fp16");
        MK(g_downsample_avg_kernel,            "downsample_avg_fp16");
        MK(g_upsample_nn_kernel,               "upsample_nn_fp16");
        MK(g_sparse_to_dense_fp16_kernel,      "sparse_to_dense_fp16");
        MK(g_gather_fp16_kernel,               "gather_fp16");
        MK(g_scatter_add_fp16_kernel,          "scatter_add_fp16");
        MK(g_half_to_float_kernel,             "half_to_float_kernel");
        MK(g_layernorm_silu_fp16_kernel,       "layernorm_silu_fp16");
        MK(g_layernorm_ss_silu_fp16_kernel,    "layernorm_scale_shift_silu_fp16");
        MK(g_residual_add_fp16_kernel,         "residual_add_fp16");
        MK(g_copy_fp16_kernel,                 "copy_fp16");
        MK(g_gn_reduce_fp32_kernel,            "gn_reduce_fp32");
        MK(g_gn_reduce_sqdev_fp32_kernel,      "gn_reduce_sqdev_fp32");
        MK(g_gn_partial_fp32_kernel,           "gn_partial_fp32");
        MK(g_gn_partial_sqdev_fp32_kernel,     "gn_partial_sqdev_fp32");
        MK(g_gn_affine_silu_fp32_kernel,       "gn_affine_silu_fp32");
        MK(g_subm_conv3d_fp32_kernel,          "subm_conv3d_fp32");
        MK(g_subm_conv3d_fp32_tiled_kernel,    "subm_conv3d_fp32_tiled");
        MK(g_linear_fp32_kernel,               "dense_linear_fp32");
        MK(g_subdivide_tile_fp32_kernel,       "subdivide_tile_fp32");
        MK(g_add_fp32_kernel,                  "add_fp32");
        #undef MK
        #undef MK_OPT
        char dev_name[256];
        clGetDeviceInfo(g_device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr);
        std::cout << "[sparse_conv_3d] Injected context: compiled kernels on: " << dev_name << std::endl;
    }
}

// ============================================================
// SLM kernel dispatch — 1D, one work-group per voxel.
// Supports slm_v2 (4 outputs/WI) and slm_v2_w8 (8 outputs/WI).
// SLM size = 27 * C_in * sizeof(half)  (must fit in device local mem ~64 KB)
// ============================================================
static void dispatch_slm_conv(
    cl_kernel kern,
    cl_mem fin,    // [N, C_in] FP16
    cl_mem nmap,   // [N, 27] I32
    cl_mem weights,// [27*C_in*C_out] FP16
    cl_mem scale,  // [C_out] FP32
    cl_mem bias,   // [C_out] FP32
    cl_mem fout,   // [N, C_out] FP16
    int N, int C_in, int C_out,
    int w_per_wi,  // 4 or 8 output channels per work-item
    int apply_relu
) {
    size_t slm_bytes = (size_t)27 * C_in * sizeof(uint16_t);
    size_t WG  = (size_t)(C_out / w_per_wi);
    if (WG < 1) WG = 1;
    size_t GWS = (size_t)N * WG;
    size_t LWS = WG;

    CL_CHECK(clSetKernelArg(kern,  0, sizeof(cl_mem), &fin));
    CL_CHECK(clSetKernelArg(kern,  1, sizeof(cl_mem), &nmap));
    CL_CHECK(clSetKernelArg(kern,  2, sizeof(cl_mem), &weights));
    CL_CHECK(clSetKernelArg(kern,  3, sizeof(cl_mem), &scale));
    CL_CHECK(clSetKernelArg(kern,  4, sizeof(cl_mem), &bias));
    CL_CHECK(clSetKernelArg(kern,  5, sizeof(cl_mem), &fout));
    CL_CHECK(clSetKernelArg(kern,  6, sizeof(int),    &N));
    CL_CHECK(clSetKernelArg(kern,  7, sizeof(int),    &C_in));
    CL_CHECK(clSetKernelArg(kern,  8, sizeof(int),    &C_out));
    CL_CHECK(clSetKernelArg(kern,  9, sizeof(int),    &apply_relu));
    CL_CHECK(clSetKernelArg(kern, 10, slm_bytes,      nullptr));  // __local half* local_feat
    CL_CHECK(clEnqueueNDRangeKernel(g_queue, kern, 1, nullptr, &GWS, &LWS, 0, nullptr, nullptr));
}

// Tiled SLM dispatch — for C_in > 1213 (e.g. 2048).
// SLM is always 27 * 1024 * sizeof(half) = 54 KB; C_in is processed in tiles.
static void dispatch_slm_tiled_conv(
    cl_mem fin, cl_mem nmap, cl_mem weights, cl_mem scale, cl_mem bias, cl_mem fout,
    int N, int C_in, int C_out, int apply_relu
) {
    constexpr size_t SLM_TILE = 1024;
    size_t slm_bytes = 27 * SLM_TILE * sizeof(uint16_t);  // 54 KB, constant
    size_t WG  = (size_t)(C_out / 8);
    if (WG < 1) WG = 1;
    size_t GWS = (size_t)N * WG;
    size_t LWS = WG;

    cl_kernel kern = g_subm_conv_fp16_slm_tiled_kernel;
    CL_CHECK(clSetKernelArg(kern,  0, sizeof(cl_mem), &fin));
    CL_CHECK(clSetKernelArg(kern,  1, sizeof(cl_mem), &nmap));
    CL_CHECK(clSetKernelArg(kern,  2, sizeof(cl_mem), &weights));
    CL_CHECK(clSetKernelArg(kern,  3, sizeof(cl_mem), &scale));
    CL_CHECK(clSetKernelArg(kern,  4, sizeof(cl_mem), &bias));
    CL_CHECK(clSetKernelArg(kern,  5, sizeof(cl_mem), &fout));
    CL_CHECK(clSetKernelArg(kern,  6, sizeof(int),    &N));
    CL_CHECK(clSetKernelArg(kern,  7, sizeof(int),    &C_in));
    CL_CHECK(clSetKernelArg(kern,  8, sizeof(int),    &C_out));
    CL_CHECK(clSetKernelArg(kern,  9, sizeof(int),    &apply_relu));
    CL_CHECK(clSetKernelArg(kern, 10, slm_bytes,      nullptr));  // __local half* local_feat
    CL_CHECK(clEnqueueNDRangeKernel(g_queue, kern, 1, nullptr, &GWS, &LWS, 0, nullptr, nullptr));
}

// SLM-K1 dispatch — one neighbor loaded per barrier iteration.
// SLM = C_in * sizeof(half) (2 KB for C_in=1024, 4 KB for C_in=2048).
// Unlike slm_v2_w8 (54 KB for C_in=1024), this tiny SLM allows ~32 WGs per
// subslice instead of ~1 — restoring full occupancy and latency hiding.
static void dispatch_slm_k1_conv(
    cl_kernel kern,  // either slm_k1 or slm_k1_f16
    cl_mem fin, cl_mem nmap, cl_mem weights, cl_mem scale, cl_mem bias, cl_mem fout,
    int N, int C_in, int C_out, int apply_relu
) {
    size_t slm_bytes = (size_t)C_in * sizeof(uint16_t);
    size_t WG  = (size_t)(C_out / 8);
    if (WG < 1) WG = 1;
    size_t GWS = (size_t)N * WG;
    size_t LWS = WG;

    CL_CHECK(clSetKernelArg(kern,  0, sizeof(cl_mem), &fin));
    CL_CHECK(clSetKernelArg(kern,  1, sizeof(cl_mem), &nmap));
    CL_CHECK(clSetKernelArg(kern,  2, sizeof(cl_mem), &weights));
    CL_CHECK(clSetKernelArg(kern,  3, sizeof(cl_mem), &scale));
    CL_CHECK(clSetKernelArg(kern,  4, sizeof(cl_mem), &bias));
    CL_CHECK(clSetKernelArg(kern,  5, sizeof(cl_mem), &fout));
    CL_CHECK(clSetKernelArg(kern,  6, sizeof(int),    &N));
    CL_CHECK(clSetKernelArg(kern,  7, sizeof(int),    &C_in));
    CL_CHECK(clSetKernelArg(kern,  8, sizeof(int),    &C_out));
    CL_CHECK(clSetKernelArg(kern,  9, sizeof(int),    &apply_relu));
    CL_CHECK(clSetKernelArg(kern, 10, slm_bytes,      nullptr));
    CL_CHECK(clEnqueueNDRangeKernel(g_queue, kern, 1, nullptr, &GWS, &LWS, 0, nullptr, nullptr));
}

// SLM-K1-INT8 dispatch — same as slm_k1_f16 but with INT8 weights.
// weights_int8: [27 * C_in * C_out] INT8 (1B each)
// w_quant_scale: [27 * C_out] FP16 — per-k per-output-channel dequantization scales
static void dispatch_slm_k1_int8_conv(
    cl_mem fin, cl_mem nmap, cl_mem weights_int8, cl_mem w_quant_scale,
    cl_mem scale, cl_mem bias, cl_mem fout,
    int N, int C_in, int C_out, int apply_relu
) {
    cl_kernel kern = g_subm_conv_fp16_slm_k1_int8_kernel;
    size_t slm_bytes = (size_t)C_in * sizeof(uint16_t);
    size_t WG  = (size_t)(C_out / 8);
    if (WG < 1) WG = 1;
    size_t GWS = (size_t)N * WG;
    size_t LWS = WG;

    CL_CHECK(clSetKernelArg(kern,  0, sizeof(cl_mem), &fin));
    CL_CHECK(clSetKernelArg(kern,  1, sizeof(cl_mem), &nmap));
    CL_CHECK(clSetKernelArg(kern,  2, sizeof(cl_mem), &weights_int8));
    CL_CHECK(clSetKernelArg(kern,  3, sizeof(cl_mem), &w_quant_scale));
    CL_CHECK(clSetKernelArg(kern,  4, sizeof(cl_mem), &scale));
    CL_CHECK(clSetKernelArg(kern,  5, sizeof(cl_mem), &bias));
    CL_CHECK(clSetKernelArg(kern,  6, sizeof(cl_mem), &fout));
    CL_CHECK(clSetKernelArg(kern,  7, sizeof(int),    &N));
    CL_CHECK(clSetKernelArg(kern,  8, sizeof(int),    &C_in));
    CL_CHECK(clSetKernelArg(kern,  9, sizeof(int),    &C_out));
    CL_CHECK(clSetKernelArg(kern, 10, sizeof(int),    &apply_relu));
    CL_CHECK(clSetKernelArg(kern, 11, slm_bytes,      nullptr));
    CL_CHECK(clEnqueueNDRangeKernel(g_queue, kern, 1, nullptr, &GWS, &LWS, 0, nullptr, nullptr));
}

// SLM-MV dispatch — multi-voxel NV=4, weight-sharing + FP16 accumulation.
// WG = NV * (C_out/8). GWS = ceil(N/NV) * WG.
// SLM = NV * C_in * sizeof(half) = 4 * 1024 * 2 = 8 KB for 1024-channel.
// Each weight element read once, reused by NV=4 voxels → 4× weight BW reduction.
static void dispatch_slm_mv_conv(
    cl_mem fin, cl_mem nmap, cl_mem weights, cl_mem scale, cl_mem bias, cl_mem fout,
    int N, int C_in, int C_out, int apply_relu
) {
    const int NV = 4;
    size_t slm_bytes = (size_t)NV * C_in * sizeof(uint16_t);  // 8 KB for C_in=1024
    size_t co_stride = (size_t)(C_out / 8);
    size_t WG        = (size_t)NV * co_stride;
    int N_groups     = (N + NV - 1) / NV;
    size_t GWS       = (size_t)N_groups * WG;
    size_t LWS       = WG;

    cl_kernel kern = g_subm_conv_fp16_slm_mv_kernel;
    CL_CHECK(clSetKernelArg(kern,  0, sizeof(cl_mem), &fin));
    CL_CHECK(clSetKernelArg(kern,  1, sizeof(cl_mem), &nmap));
    CL_CHECK(clSetKernelArg(kern,  2, sizeof(cl_mem), &weights));
    CL_CHECK(clSetKernelArg(kern,  3, sizeof(cl_mem), &scale));
    CL_CHECK(clSetKernelArg(kern,  4, sizeof(cl_mem), &bias));
    CL_CHECK(clSetKernelArg(kern,  5, sizeof(cl_mem), &fout));
    CL_CHECK(clSetKernelArg(kern,  6, sizeof(int),    &N));
    CL_CHECK(clSetKernelArg(kern,  7, sizeof(int),    &C_in));
    CL_CHECK(clSetKernelArg(kern,  8, sizeof(int),    &C_out));
    CL_CHECK(clSetKernelArg(kern,  9, sizeof(int),    &apply_relu));
    CL_CHECK(clSetKernelArg(kern, 10, slm_bytes,      nullptr));
    CL_CHECK(clEnqueueNDRangeKernel(g_queue, kern, 1, nullptr, &GWS, &LWS, 0, nullptr, nullptr));
}

// SLM-VTILE dispatch — in-register weight reuse across VT=4 voxels per thread.
// WG = C_out/8 threads. GWS = ceil(N/VT) * (C_out/8).
// SLM = VT * C_in * sizeof(half). Each weight vload8 reused across VT accumulators.
static void dispatch_slm_vtile_conv(
    cl_mem fin, cl_mem nmap, cl_mem weights, cl_mem scale, cl_mem bias, cl_mem fout,
    int N, int C_in, int C_out, int apply_relu
) {
    const int VT = 4;
    size_t slm_bytes = (size_t)VT * C_in * sizeof(uint16_t);
    size_t WG  = (size_t)(C_out / 8);
    if (WG < 1) WG = 1;
    int N_groups = (N + VT - 1) / VT;
    size_t GWS = (size_t)N_groups * WG;
    size_t LWS = WG;

    cl_kernel kern = g_subm_conv_fp16_slm_vtile_kernel;
    CL_CHECK(clSetKernelArg(kern,  0, sizeof(cl_mem), &fin));
    CL_CHECK(clSetKernelArg(kern,  1, sizeof(cl_mem), &nmap));
    CL_CHECK(clSetKernelArg(kern,  2, sizeof(cl_mem), &weights));
    CL_CHECK(clSetKernelArg(kern,  3, sizeof(cl_mem), &scale));
    CL_CHECK(clSetKernelArg(kern,  4, sizeof(cl_mem), &bias));
    CL_CHECK(clSetKernelArg(kern,  5, sizeof(cl_mem), &fout));
    CL_CHECK(clSetKernelArg(kern,  6, sizeof(int),    &N));
    CL_CHECK(clSetKernelArg(kern,  7, sizeof(int),    &C_in));
    CL_CHECK(clSetKernelArg(kern,  8, sizeof(int),    &C_out));
    CL_CHECK(clSetKernelArg(kern,  9, sizeof(int),    &apply_relu));
    CL_CHECK(clSetKernelArg(kern, 10, slm_bytes,      nullptr));
    CL_CHECK(clEnqueueNDRangeKernel(g_queue, kern, 1, nullptr, &GWS, &LWS, 0, nullptr, nullptr));
}

// Grow a persistent GPU buffer to at least `need` bytes
static void ensure_cl_buf(cl_mem& buf, size_t& buf_sz, size_t need,                           cl_mem_flags flags = CL_MEM_READ_WRITE) {
    if (need <= buf_sz) return;
    cl_int err;
    if (buf) { clReleaseMemObject(buf); buf = nullptr; }
    buf = clCreateBuffer(g_context, flags, need, nullptr, &err);
    if (err != CL_SUCCESS)
        throw std::runtime_error("[sparse_conv_3d] clCreateBuffer failed: " +
                                 std::to_string(err));
    buf_sz = need;
}

// GPU SubMConv3d with FP32 I/O — Gather-Linear-Scatter approach.
//
// Instead of per-work-item accumulation across all 27 neighbors (poor cache
// utilization, ~0.4% of peak throughput), this function:
//   1. Uploads FP32 features → GPU, converts to FP16
//   2. For each of 27 kernel positions:
//      a. Gather: copy features from neighbor indices → dense matrix
//      b. Linear: gathered × W_k → partial output (reuses linear_fp16 kernel)
//      c. Scatter-add: accumulate partial output to result
//   3. Apply BN (scale*x + bias)
//   4. Downloads FP16 → FP32
//
// This uses the linear_fp16 kernel which has better memory access patterns
// (coalesced reads, no indirect addressing in the inner loop).
// Weights are cached on GPU across calls.
// ReLU is NOT applied here — caller must apply it after.

// Persistent GPU buffers for gather-linear-scatter
static cl_mem  g_gathered_buf     = nullptr;
static size_t  g_gathered_buf_sz  = 0;
static cl_mem  g_partial_buf      = nullptr;
static size_t  g_partial_buf_sz   = 0;
static cl_mem  g_out_f32_buf      = nullptr;
static size_t  g_out_f32_buf_sz   = 0;

static void gpu_subm_conv3d_full(
    const float*   in_f32,  // [N, C_in]  FP32
    const float*   w_f32,   // [27*C_in*C_out] FP32  (stable pointer for caching)
    const float*   s_f32,   // [C_out] FP32 (BN scale)
    const float*   b_f32,   // [C_out] FP32 (BN bias)
    const int32_t* nmap,    // [N, 27]  I32
    int N, int C_in, int C_out,
    float* out_f32          // [N, C_out] FP32
) {
    size_t in_f16_sz  = (size_t)N * C_in  * 2u;
    size_t out_f16_sz = (size_t)N * C_out * 2u;
    size_t nmap_sz    = (size_t)N * 27    * 4u;
    size_t in_f32_sz  = (size_t)N * C_in  * 4u;
    size_t out_f32_sz = (size_t)N * C_out * 4u;
    size_t gathered_sz = (size_t)N * C_in * 2u;   // [N, C_in] FP16
    size_t partial_sz  = (size_t)N * C_out * 2u;  // [N, C_out] FP16

    // Ensure GPU buffers
    ensure_cl_buf(g_feat_buf[0], g_feat_buf_sz[0], in_f16_sz);      // input FP16
    ensure_cl_buf(g_feat_buf[1], g_feat_buf_sz[1], out_f16_sz);     // output FP16
    ensure_cl_buf(g_neighbor_map_buf, g_nmb_sz, nmap_sz, CL_MEM_READ_ONLY);
    ensure_cl_buf(g_feat_fp32_buf, g_feat_fp32_buf_sz, in_f32_sz);  // input FP32
    ensure_cl_buf(g_gathered_buf, g_gathered_buf_sz, gathered_sz);    // gathered FP16
    ensure_cl_buf(g_partial_buf, g_partial_buf_sz, partial_sz);       // partial FP16
    ensure_cl_buf(g_out_f32_buf, g_out_f32_buf_sz, out_f32_sz);     // output FP32

    // ── Weight caching: convert + upload once per unique weight pointer ──
    // Weights are split per kernel position: 27 × [C_in, C_out] FP16 matrices
    auto wit = g_weight_cache.find(w_f32);
    if (wit == g_weight_cache.end()) {
        CachedWeights cw;
        size_t n_w = (size_t)27 * C_in * C_out;

        // ── FP16 weights (always computed) ──
        size_t w_f16_sz = n_w * 2u;
        std::vector<uint16_t> w_f16(n_w);
        for (size_t i = 0; i < n_w; i++) w_f16[i] = f32_to_f16(w_f32[i]);

        cl_int e;
        cw.w_bytes = w_f16_sz;
        cw.weights_fp16 = clCreateBuffer(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          w_f16_sz, w_f16.data(), &e);
        CL_CHECK(e);
        size_t s_sz = (size_t)C_out * 4u;
        cw.s_bytes = s_sz;
        cw.scale = clCreateBuffer(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   s_sz, (void*)s_f32, &e);
        CL_CHECK(e);
        cw.bias = clCreateBuffer(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  s_sz, (void*)b_f32, &e);
        CL_CHECK(e);

        // ── INT8 weight quantization (computed in C++, no export changes needed) ──
        // Per-kernel-position per-output-channel symmetric quantization:
        //   scale[k][co] = max(|w[k, :, co]|) / 127
        //   w_int8[k][ci][co] = clip(round(w[k][ci][co] / scale[k][co]), -127, 127)
        // Scale applied once per k outside the ci loop in the INT8 kernel.
        if (g_subm_conv_fp16_slm_k1_int8_kernel && (C_in > 256) && (C_out >= 8)) {
            std::vector<int8_t>    w_int8_cpu(n_w);
            std::vector<uint16_t>  w_quant_scale_cpu((size_t)27 * C_out);

            // Pass 1: find max|w| per (k, co)
            std::vector<float> max_abs((size_t)27 * C_out, 1e-12f);
            for (int k = 0; k < 27; k++)
                for (int ci = 0; ci < C_in; ci++)
                    for (int co = 0; co < C_out; co++) {
                        float abw = fabsf(w_f32[(size_t)k*C_in*C_out + (size_t)ci*C_out + co]);
                        if (abw > max_abs[k*C_out + co]) max_abs[k*C_out + co] = abw;
                    }

            // Compute scales (FP16) and inverse scales for quantization
            std::vector<float> inv_sc((size_t)27 * C_out);
            for (int kco = 0; kco < 27*C_out; kco++) {
                float s = max_abs[kco] / 127.0f;
                w_quant_scale_cpu[kco] = f32_to_f16(s);
                inv_sc[kco] = 1.0f / s;
            }

            // Pass 2: quantize
            for (int k = 0; k < 27; k++)
                for (int ci = 0; ci < C_in; ci++)
                    for (int co = 0; co < C_out; co++) {
                        float wval = w_f32[(size_t)k*C_in*C_out + (size_t)ci*C_out + co];
                        int qi = (int)roundf(wval * inv_sc[k*C_out + co]);
                        if (qi > 127) qi = 127; if (qi < -127) qi = -127;
                        w_int8_cpu[(size_t)k*C_in*C_out + (size_t)ci*C_out + co] = (int8_t)qi;
                    }

            cw.w_int8_bytes = n_w;
            cw.weights_int8 = clCreateBuffer(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              n_w, w_int8_cpu.data(), &e);
            CL_CHECK(e);
            size_t wqs_sz = (size_t)27 * C_out * 2u;
            cw.w_quant_scale = clCreateBuffer(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                               wqs_sz, w_quant_scale_cpu.data(), &e);
            CL_CHECK(e);
            cw.has_int8 = true;
        }

        g_weight_cache[w_f32] = cw;
        wit = g_weight_cache.find(w_f32);
    }
    CachedWeights& cw = wit->second;

    // ── Upload features: FP32 to GPU, convert to FP16 ──
    CL_CHECK(clEnqueueWriteBuffer(g_queue, g_feat_fp32_buf,    CL_FALSE, 0, in_f32_sz, in_f32, 0, nullptr, nullptr));
    CL_CHECK(clEnqueueWriteBuffer(g_queue, g_neighbor_map_buf, CL_TRUE,  0, nmap_sz,   nmap,   0, nullptr, nullptr));
    // This path overwrote g_neighbor_map_buf with its own nmap → invalidate the fused cache.
    g_gpu_nmap_valid = false;

    // GPU FP32→FP16 conversion
    {
        int n_el = N * C_in;
        CL_CHECK(clSetKernelArg(g_float_to_half_kernel, 0, sizeof(cl_mem), &g_feat_fp32_buf));
        CL_CHECK(clSetKernelArg(g_float_to_half_kernel, 1, sizeof(cl_mem), &g_feat_buf[0]));
        CL_CHECK(clSetKernelArg(g_float_to_half_kernel, 2, sizeof(int), &n_el));
        size_t gws = (size_t)((n_el + 255) / 256) * 256;
        size_t lws = 256;
        CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_float_to_half_kernel, 1, nullptr,
                                         &gws, &lws, 0, nullptr, nullptr));
    }

    // ── Zero the output buffer ──
    {
        uint16_t zval = 0;  // FP16 zero
        CL_CHECK(clEnqueueFillBuffer(g_queue, g_feat_buf[1], &zval, sizeof(uint16_t),
                                      0, out_f16_sz, 0, nullptr, nullptr));
    }

    // ── Convolution: SLM path (single fused kernel) or fallback gather-linear-scatter ──
    //
    // Kernel selection (in priority order):
    //
    //  slm_k1:     C_in > 256, C_out >= 8
    //    Loads ONE neighbor per barrier iteration → SLM = C_in*2B (2–4 KB for 1024/2048 ch)
    //    ~32 concurrent WGs per 64 KB subslice → full occupancy, good latency hiding.
    //    Replaces slm_direct for large C_in where 27*C_in SLM kills occupancy.
    //
    //  slm_mv:     C_in > 256, C_out >= 64 — NV=4 voxels share weight loads
    //    4× weight BW reduction for weight-bandwidth-bound cases (e.g. 1024→1024).
    //    SLM = NV*C_in*2B: 8 KB (C_in=1024), 16 KB (C_in=2048) → 4-8 WGs/subslice.
    //    Register pressure fixed: nb[27] per thread (was nb[NV][27]=108 ints → spilling).
    //
    //  slm_direct: C_in <= 256, C_out/8 >= 32
    //    Loads all 27 neighbors into SLM = 27*C_in*2B (≤ 13.5 KB for 128 channels)
    //    Very low SLM usage → excellent occupancy. Best for small C_in.
    //
    //  slm_tiled:  256 < C_in <= 2048, C_out/8 >= 64  (kept for reference; usually slm_k1 wins)
    //    Tiles C_in in 1024-wide chunks.  SLM = 27*1024*2 = 54 KB — poor occupancy.
    //    Only used as fallback if slm_k1 kernel unavailable.
    //
    //  fallback:   gather-linear-scatter (any C_in, but much lower throughput)
    //
    const size_t slm_need = (size_t)27 * C_in * sizeof(uint16_t);

    // slm_mv: disabled — nb[27] per thread + re-reading neighbor_map in Phase 1
    // causes more global memory traffic than the weight-BW benefit provides.
    // slm_k1_f16 remains the primary large-C_in kernel.
    const bool slm_mv = false;

    // slm_k1_int8: INT8 weights reduce BW by 2× for weight-BW-bound large-C_in convs.
    // Requires INT8 weights to have been computed during weight cache population.
    const bool slm_k1_int8 = !slm_mv && (C_in > 256) && (C_out >= 8)
                              && cw.has_int8 && g_subm_conv_fp16_slm_k1_int8_kernel;

    // slm_k1_f16: per-neighbor SLM + FP16 inner accumulation + 4-channel unrolling.
    // FP16 gives 2× compute throughput vs FP32; 4-unroll hides weight-load latency.
    const bool slm_k1_f16 = !slm_mv && !slm_k1_int8 && (C_in > 256) && (C_out >= 8)
                             && g_subm_conv_fp16_slm_k1_f16_kernel;

    // slm_k1: per-neighbor SLM, FP32 accumulation. Fallback when f16 unavailable.
    const bool slm_k1 = !slm_mv && !slm_k1_int8 && !slm_k1_f16 && (C_in > 256) && (C_out >= 8)
                        && g_subm_conv_fp16_slm_k1_kernel;

    // slm_direct: full 27-neighbor SLM, ideal for small C_in (<= 256)
    // WG = C_out/8; require >= 32 threads so each thread loads <= 27 elements.
    const bool slm_direct = !slm_mv && !slm_k1_int8 && !slm_k1_f16 && !slm_k1
                             && (slm_need <= 65536) && (C_out / 8 >= 8)
                             && g_subm_conv_fp16_slm_v2_w8_kernel;

    // Tiled SLM: fallback when slm_k1 unavailable but slm_direct doesn't fit.
    // Require WG >= 64 to keep per-thread load count acceptable.
    const bool slm_tiled  = !slm_mv && !slm_k1_int8 && !slm_k1_f16 && !slm_k1 && !slm_direct
                             && (C_in <= 2048) && (C_out / 8 >= 64)
                             && g_subm_conv_fp16_slm_tiled_kernel;

    if (slm_mv) {
        // NV=4 weight sharing: 4× weight BW reduction, best for 1024→1024
        dispatch_slm_mv_conv(g_feat_buf[0], g_neighbor_map_buf, cw.weights_fp16,
                             cw.scale, cw.bias, g_feat_buf[1],
                             N, C_in, C_out, /*apply_relu=*/0);
    } else if (slm_k1_int8) {
        // Best path: INT8 weights (2× less BW) + SLM input reads + FP16 accumulation
        dispatch_slm_k1_int8_conv(g_feat_buf[0], g_neighbor_map_buf,
                                   cw.weights_int8, cw.w_quant_scale,
                                   cw.scale, cw.bias, g_feat_buf[1],
                                   N, C_in, C_out, /*apply_relu=*/0);
    } else if (slm_k1_f16) {
        // SLM input reads + FP16 accumulation + 4-unroll
        dispatch_slm_k1_conv(g_subm_conv_fp16_slm_k1_f16_kernel,
                             g_feat_buf[0], g_neighbor_map_buf, cw.weights_fp16,
                             cw.scale, cw.bias, g_feat_buf[1],
                             N, C_in, C_out, /*apply_relu=*/0);
    } else if (slm_k1) {
        // SLM input reads + FP32 accumulation (fallback)
        dispatch_slm_k1_conv(g_subm_conv_fp16_slm_k1_kernel,
                             g_feat_buf[0], g_neighbor_map_buf, cw.weights_fp16,
                             cw.scale, cw.bias, g_feat_buf[1],
                             N, C_in, C_out, /*apply_relu=*/0);
    } else if (slm_direct) {
        // Single SLM kernel: fuses neighbor gather + conv + BN
        // apply_relu=0 — BN is applied in kernel, ReLU applied by caller on CPU
        dispatch_slm_conv(g_subm_conv_fp16_slm_v2_w8_kernel,
                          g_feat_buf[0], g_neighbor_map_buf, cw.weights_fp16,
                          cw.scale, cw.bias, g_feat_buf[1],
                          N, C_in, C_out, /*w_per_wi=*/8, /*apply_relu=*/0);
    } else if (slm_tiled) {
        // Tiled SLM: processes C_in in 1024-wide tiles, each fitting in 64 KB
        dispatch_slm_tiled_conv(g_feat_buf[0], g_neighbor_map_buf, cw.weights_fp16,
                                 cw.scale, cw.bias, g_feat_buf[1],
                                 N, C_in, C_out, /*apply_relu=*/0);
    } else {
        // Fallback: 27×gather-linear-scatter (handles any C_in)
        for (int k = 0; k < 27; k++) {
            size_t w_offset = (size_t)k * C_in * C_out * 2u;

            // 1. Gather
            {
                CL_CHECK(clSetKernelArg(g_gather_fp16_kernel, 0, sizeof(cl_mem), &g_feat_buf[0]));
                CL_CHECK(clSetKernelArg(g_gather_fp16_kernel, 1, sizeof(cl_mem), &g_neighbor_map_buf));
                CL_CHECK(clSetKernelArg(g_gather_fp16_kernel, 2, sizeof(int), &N));
                CL_CHECK(clSetKernelArg(g_gather_fp16_kernel, 3, sizeof(int), &C_in));
                CL_CHECK(clSetKernelArg(g_gather_fp16_kernel, 4, sizeof(int), &k));
                CL_CHECK(clSetKernelArg(g_gather_fp16_kernel, 5, sizeof(cl_mem), &g_gathered_buf));
                size_t gws[2] = {(size_t)N, (size_t)C_in};
                size_t lws[2] = {std::min((size_t)N, (size_t)64), std::min((size_t)C_in, (size_t)16)};
                CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_gather_fp16_kernel, 2, nullptr,
                                                 gws, lws, 0, nullptr, nullptr));
            }

            // 2. Linear: partial = gathered × W_k
            {
                cl_buffer_region region = {w_offset, (size_t)C_in * C_out * 2u};
                cl_int sbe;
                cl_mem w_k_buf = clCreateSubBuffer(cw.weights_fp16, CL_MEM_READ_ONLY,
                                                    CL_BUFFER_CREATE_TYPE_REGION, &region, &sbe);
                CL_CHECK(sbe);
                int has_bias_flag = 0;
                CL_CHECK(clSetKernelArg(g_linear_fp16_kernel, 0, sizeof(cl_mem), &g_gathered_buf));
                CL_CHECK(clSetKernelArg(g_linear_fp16_kernel, 1, sizeof(cl_mem), &w_k_buf));
                CL_CHECK(clSetKernelArg(g_linear_fp16_kernel, 2, sizeof(cl_mem), &cw.bias));
                CL_CHECK(clSetKernelArg(g_linear_fp16_kernel, 3, sizeof(int), &N));
                CL_CHECK(clSetKernelArg(g_linear_fp16_kernel, 4, sizeof(int), &C_in));
                CL_CHECK(clSetKernelArg(g_linear_fp16_kernel, 5, sizeof(int), &C_out));
                CL_CHECK(clSetKernelArg(g_linear_fp16_kernel, 6, sizeof(cl_mem), &g_partial_buf));
                CL_CHECK(clSetKernelArg(g_linear_fp16_kernel, 7, sizeof(int), &has_bias_flag));
                size_t gws[2] = {(size_t)N, (size_t)C_out};
                size_t lws[2] = {std::min((size_t)N, (size_t)64), std::min((size_t)C_out, (size_t)16)};
                CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_linear_fp16_kernel, 2, nullptr,
                                                 gws, lws, 0, nullptr, nullptr));
                clReleaseMemObject(w_k_buf);
            }

            // 3. Scatter-add: output += partial
            {
                int total = N * C_out;
                CL_CHECK(clSetKernelArg(g_scatter_add_fp16_kernel, 0, sizeof(cl_mem), &g_feat_buf[1]));
                CL_CHECK(clSetKernelArg(g_scatter_add_fp16_kernel, 1, sizeof(cl_mem), &g_partial_buf));
                CL_CHECK(clSetKernelArg(g_scatter_add_fp16_kernel, 2, sizeof(int), &total));
                size_t gws = (size_t)((total + 255) / 256) * 256;
                size_t lws = 256;
                CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_scatter_add_fp16_kernel, 1, nullptr,
                                                 &gws, &lws, 0, nullptr, nullptr));
            }
        }

        // Apply BN separately (not fused — only needed for fallback path)
        {
            int relu_flag = 1;
            CL_CHECK(clSetKernelArg(g_apply_bn_relu_kernel, 0, sizeof(cl_mem), &g_feat_buf[1]));
            CL_CHECK(clSetKernelArg(g_apply_bn_relu_kernel, 1, sizeof(cl_mem), &cw.scale));
            CL_CHECK(clSetKernelArg(g_apply_bn_relu_kernel, 2, sizeof(cl_mem), &cw.bias));
            CL_CHECK(clSetKernelArg(g_apply_bn_relu_kernel, 3, sizeof(int), &N));
            CL_CHECK(clSetKernelArg(g_apply_bn_relu_kernel, 4, sizeof(int), &C_out));
            CL_CHECK(clSetKernelArg(g_apply_bn_relu_kernel, 5, sizeof(int), &relu_flag));
            size_t gws[2] = {(size_t)N, (size_t)C_out};
            size_t lws[2] = {std::min((size_t)N, (size_t)64), std::min((size_t)C_out, (size_t)16)};
            CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_apply_bn_relu_kernel, 2, nullptr,
                                             gws, lws, 0, nullptr, nullptr));
        }
    }

    // ── Download: FP16 → FP32 on GPU, then read ──
    {
        int n_el = N * C_out;
        CL_CHECK(clSetKernelArg(g_half_to_float_kernel, 0, sizeof(cl_mem), &g_feat_buf[1]));
        CL_CHECK(clSetKernelArg(g_half_to_float_kernel, 1, sizeof(cl_mem), &g_out_f32_buf));
        CL_CHECK(clSetKernelArg(g_half_to_float_kernel, 2, sizeof(int), &n_el));
        size_t gws = (size_t)((n_el + 255) / 256) * 256;
        size_t lws = 256;
        CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_half_to_float_kernel, 1, nullptr,
                                         &gws, &lws, 0, nullptr, nullptr));
    }
    CL_CHECK(clEnqueueReadBuffer(g_queue, g_out_f32_buf, CL_TRUE, 0,
                                  out_f32_sz, out_f32, 0, nullptr, nullptr));
}

// ═══════════════════════════════════════════════════════════════════════════════
// GPU-only SubMConv3d — assumes input ALREADY in g_feat_buf[src_idx] as FP16.
// Output goes to g_feat_buf[dst_idx] as FP16 (no BN, no download).
// Weight caching reused from gpu_subm_conv3d_full.
// ═══════════════════════════════════════════════════════════════════════════════

static void gpu_subm_conv3d_inplace(
    const float*   w_f32,   // [27*C_in*C_out] FP32 (stable pointer for caching)
    const float*   s_f32,   // [C_out] FP32 (BN scale — ones for SAM3D conv)
    const float*   b_f32,   // [C_out] FP32 (BN bias — conv bias for SAM3D)
    int N, int C_in, int C_out,
    int src_idx, int dst_idx  // indices into g_feat_buf[]
) {
    size_t out_f16_sz = (size_t)N * C_out * 2u;
    ensure_cl_buf(g_feat_buf[dst_idx], g_feat_buf_sz[dst_idx], out_f16_sz);

    // Zero output buffer
    uint16_t zval = 0;
    CL_CHECK(clEnqueueFillBuffer(g_queue, g_feat_buf[dst_idx], &zval, sizeof(uint16_t),
                                  0, out_f16_sz, 0, nullptr, nullptr));

    // Get or cache weights
    auto wit = g_weight_cache.find(w_f32);
    if (wit == g_weight_cache.end()) {
        CachedWeights cw;
        size_t n_w = (size_t)27 * C_in * C_out;
        size_t w_f16_sz = n_w * 2u;
        std::vector<uint16_t> w_f16(n_w);
        for (size_t i = 0; i < n_w; i++) w_f16[i] = f32_to_f16(w_f32[i]);
        cl_int e;
        cw.w_bytes = w_f16_sz;
        cw.weights_fp16 = clCreateBuffer(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          w_f16_sz, w_f16.data(), &e); CL_CHECK(e);
        size_t s_sz = (size_t)C_out * 4u;
        cw.s_bytes = s_sz;
        cw.scale = clCreateBuffer(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   s_sz, (void*)s_f32, &e); CL_CHECK(e);
        cw.bias = clCreateBuffer(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  s_sz, (void*)b_f32, &e); CL_CHECK(e);
        cw.has_int8 = false;  // skip INT8 for inplace path
        g_weight_cache[w_f32] = cw;
        wit = g_weight_cache.find(w_f32);
    }
    CachedWeights& cw = wit->second;

    // Dispatch conv kernel (same logic as gpu_subm_conv3d_full)
    const bool slm_k1_int8 = false;
    // slm_vtile: in-register weight reuse across VT=4 voxels. Env-gated A/B test.
    // SLM = VT*C_in*2B must fit (<= 64 KB) → C_in <= 8192. Reuses weight loads
    // VT× to cut weight-BW-bound convs. Only beneficial when C_out is large
    // enough for good WG occupancy (WG = C_out/8 threads): C_out >= 512 → WG >= 64.
    // Skinny convs (C_out=128 → WG=16) have too little parallelism and regress.
    static const bool s_vtile = (std::getenv("SAM3D_VTILE") != nullptr);
    const size_t vtile_slm = (size_t)4 * C_in * sizeof(uint16_t);
    const bool slm_vtile = s_vtile && (C_in > 256) && (C_out >= 512) && (C_out % 8 == 0)
                            && (vtile_slm <= 65536) && g_subm_conv_fp16_slm_vtile_kernel;
    const bool slm_k1_f16 = !slm_vtile && (C_in > 256) && (C_out >= 8) && g_subm_conv_fp16_slm_k1_f16_kernel;
    const bool slm_k1 = !slm_vtile && !slm_k1_f16 && (C_in > 256) && (C_out >= 8) && g_subm_conv_fp16_slm_k1_kernel;
    const size_t slm_need = (size_t)27 * C_in * sizeof(uint16_t);
    const bool slm_direct = !slm_vtile && !slm_k1_f16 && !slm_k1 && (slm_need <= 65536) && (C_out / 8 >= 8)
                             && g_subm_conv_fp16_slm_v2_w8_kernel;

    if (slm_vtile) {
        dispatch_slm_vtile_conv(g_feat_buf[src_idx], g_neighbor_map_buf, cw.weights_fp16,
                                cw.scale, cw.bias, g_feat_buf[dst_idx],
                                N, C_in, C_out, /*apply_relu=*/0);
    } else if (slm_k1_f16) {
        dispatch_slm_k1_conv(g_subm_conv_fp16_slm_k1_f16_kernel,
                             g_feat_buf[src_idx], g_neighbor_map_buf, cw.weights_fp16,
                             cw.scale, cw.bias, g_feat_buf[dst_idx],
                             N, C_in, C_out, /*apply_relu=*/0);
    } else if (slm_k1) {
        dispatch_slm_k1_conv(g_subm_conv_fp16_slm_k1_kernel,
                             g_feat_buf[src_idx], g_neighbor_map_buf, cw.weights_fp16,
                             cw.scale, cw.bias, g_feat_buf[dst_idx],
                             N, C_in, C_out, /*apply_relu=*/0);
    } else if (slm_direct) {
        dispatch_slm_conv(g_subm_conv_fp16_slm_v2_w8_kernel,
                          g_feat_buf[src_idx], g_neighbor_map_buf, cw.weights_fp16,
                          cw.scale, cw.bias, g_feat_buf[dst_idx],
                          N, C_in, C_out, /*w_per_wi=*/8, /*apply_relu=*/0);
    } else {
        // Fallback: gather-linear-scatter
        size_t gathered_sz = (size_t)N * C_in * 2u;
        size_t partial_sz  = (size_t)N * C_out * 2u;
        ensure_cl_buf(g_gathered_buf, g_gathered_buf_sz, gathered_sz);
        ensure_cl_buf(g_partial_buf, g_partial_buf_sz, partial_sz);
        for (int k = 0; k < 27; k++) {
            size_t w_offset = (size_t)k * C_in * C_out * 2u;
            {
                CL_CHECK(clSetKernelArg(g_gather_fp16_kernel, 0, sizeof(cl_mem), &g_feat_buf[src_idx]));
                CL_CHECK(clSetKernelArg(g_gather_fp16_kernel, 1, sizeof(cl_mem), &g_neighbor_map_buf));
                CL_CHECK(clSetKernelArg(g_gather_fp16_kernel, 2, sizeof(int), &N));
                CL_CHECK(clSetKernelArg(g_gather_fp16_kernel, 3, sizeof(int), &C_in));
                CL_CHECK(clSetKernelArg(g_gather_fp16_kernel, 4, sizeof(int), &k));
                CL_CHECK(clSetKernelArg(g_gather_fp16_kernel, 5, sizeof(cl_mem), &g_gathered_buf));
                size_t gws[2] = {(size_t)N, (size_t)C_in};
                size_t lws[2] = {std::min((size_t)N, (size_t)64), std::min((size_t)C_in, (size_t)16)};
                CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_gather_fp16_kernel, 2, nullptr,
                                                 gws, lws, 0, nullptr, nullptr));
            }
            {
                cl_buffer_region region = {w_offset, (size_t)C_in * C_out * 2u};
                cl_int sbe;
                cl_mem w_k_buf = clCreateSubBuffer(cw.weights_fp16, CL_MEM_READ_ONLY,
                                                    CL_BUFFER_CREATE_TYPE_REGION, &region, &sbe); CL_CHECK(sbe);
                int has_bias_flag = 0;
                CL_CHECK(clSetKernelArg(g_linear_fp16_kernel, 0, sizeof(cl_mem), &g_gathered_buf));
                CL_CHECK(clSetKernelArg(g_linear_fp16_kernel, 1, sizeof(cl_mem), &w_k_buf));
                CL_CHECK(clSetKernelArg(g_linear_fp16_kernel, 2, sizeof(cl_mem), &cw.bias));
                CL_CHECK(clSetKernelArg(g_linear_fp16_kernel, 3, sizeof(int), &N));
                CL_CHECK(clSetKernelArg(g_linear_fp16_kernel, 4, sizeof(int), &C_in));
                CL_CHECK(clSetKernelArg(g_linear_fp16_kernel, 5, sizeof(int), &C_out));
                CL_CHECK(clSetKernelArg(g_linear_fp16_kernel, 6, sizeof(cl_mem), &g_partial_buf));
                CL_CHECK(clSetKernelArg(g_linear_fp16_kernel, 7, sizeof(int), &has_bias_flag));
                size_t gws[2] = {(size_t)N, (size_t)C_out};
                size_t lws[2] = {std::min((size_t)N, (size_t)64), std::min((size_t)C_out, (size_t)16)};
                CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_linear_fp16_kernel, 2, nullptr,
                                                 gws, lws, 0, nullptr, nullptr));
                clReleaseMemObject(w_k_buf);
            }
            {
                int total = N * C_out;
                CL_CHECK(clSetKernelArg(g_scatter_add_fp16_kernel, 0, sizeof(cl_mem), &g_feat_buf[dst_idx]));
                CL_CHECK(clSetKernelArg(g_scatter_add_fp16_kernel, 1, sizeof(cl_mem), &g_partial_buf));
                CL_CHECK(clSetKernelArg(g_scatter_add_fp16_kernel, 2, sizeof(int), &total));
                size_t gws = (size_t)((total + 255) / 256) * 256;
                size_t lws = 256;
                CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_scatter_add_fp16_kernel, 1, nullptr,
                                                 &gws, &lws, 0, nullptr, nullptr));
            }
        }
        // Apply BN (no ReLU for inplace path — activation handled by fused norm)
        {
            int relu_flag = 0;
            CL_CHECK(clSetKernelArg(g_apply_bn_relu_kernel, 0, sizeof(cl_mem), &g_feat_buf[dst_idx]));
            CL_CHECK(clSetKernelArg(g_apply_bn_relu_kernel, 1, sizeof(cl_mem), &cw.scale));
            CL_CHECK(clSetKernelArg(g_apply_bn_relu_kernel, 2, sizeof(cl_mem), &cw.bias));
            CL_CHECK(clSetKernelArg(g_apply_bn_relu_kernel, 3, sizeof(int), &N));
            CL_CHECK(clSetKernelArg(g_apply_bn_relu_kernel, 4, sizeof(int), &C_out));
            CL_CHECK(clSetKernelArg(g_apply_bn_relu_kernel, 5, sizeof(int), &relu_flag));
            size_t gws[2] = {(size_t)N, (size_t)C_out};
            size_t lws[2] = {std::min((size_t)N, (size_t)64), std::min((size_t)C_out, (size_t)16)};
            CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_apply_bn_relu_kernel, 2, nullptr,
                                             gws, lws, 0, nullptr, nullptr));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GPU Fused Resblock — entire SparseResBlock3d on GPU, single upload/download.
//
// Flow: upload → copy_identity → norm1+silu → conv1 → norm2_act → conv2 →
//       skip_linear(identity) → residual_add → download
//
// All intermediate data stays in GPU FP16 buffers. Eliminates 5 PCIe round-trips
// per resblock call (norm1, conv1_download, conv2_upload, norm2, skip).
// ═══════════════════════════════════════════════════════════════════════════════

static void gpu_fused_resblock(
    const float*   in_f32,        // [N, C_in]  FP32 input features
    const int32_t* nmap,          // [N, 27]  I32 neighbor map
    int N, int C_in, int C_out,   // C_out = conv output channels (may differ from C_in)
    // Conv1 weights: [27*C_in*C_mid] — C_mid = C_out for SAM3D
    const float* conv1_w, const float* conv1_s, const float* conv1_b, int C_mid,
    // Conv2 weights: [27*C_mid*C_out]
    const float* conv2_w, const float* conv2_s, const float* conv2_b,
    // Norm1 params: gamma[C_in], beta[C_in]
    const float* norm1_gamma, const float* norm1_beta,
    // Norm2_act params: emb_scale[C_mid], emb_shift[C_mid]
    const float* emb_scale, const float* emb_shift,
    // Skip: if C_in != C_out, skip_w[C_in*C_out], skip_b[C_out]; else nullptr
    const float* skip_w, const float* skip_b,
    float* out_f32,               // [N, C_out] FP32 output
    bool nmap_host_cached         // true => host nmap unchanged since last call (coords stable)
) {
    size_t in_f16_sz   = (size_t)N * C_in  * 2u;
    size_t mid_f16_sz  = (size_t)N * C_mid * 2u;
    size_t out_f16_sz  = (size_t)N * C_out * 2u;
    size_t nmap_sz     = (size_t)N * 27    * 4u;
    size_t in_f32_sz   = (size_t)N * C_in  * 4u;
    size_t out_f32_sz  = (size_t)N * C_out * 4u;

    // Ensure buffers: [0]=input/work, [1]=output/work, [2]=identity, [3]=skip_result
    size_t max_f16 = std::max({in_f16_sz, mid_f16_sz, out_f16_sz});
    ensure_cl_buf(g_feat_buf[0], g_feat_buf_sz[0], max_f16);
    ensure_cl_buf(g_feat_buf[1], g_feat_buf_sz[1], max_f16);
    ensure_cl_buf(g_identity_buf, g_identity_buf_sz, in_f16_sz);
    ensure_cl_buf(g_feat_fp32_buf, g_feat_fp32_buf_sz, std::max(in_f32_sz, out_f32_sz));
    ensure_cl_buf(g_neighbor_map_buf, g_nmb_sz, nmap_sz, CL_MEM_READ_ONLY);
    ensure_cl_buf(g_out_f32_buf, g_out_f32_buf_sz, out_f32_sz);

    // Upload emb_scale/shift to GPU
    size_t emb_sz = (size_t)C_mid * 4u;
    ensure_cl_buf(g_emb_scale_buf, g_emb_scale_buf_sz, emb_sz);
    ensure_cl_buf(g_emb_shift_buf, g_emb_shift_buf_sz, emb_sz);
    CL_CHECK(clEnqueueWriteBuffer(g_queue, g_emb_scale_buf, CL_FALSE, 0, emb_sz, emb_scale, 0, nullptr, nullptr));
    CL_CHECK(clEnqueueWriteBuffer(g_queue, g_emb_shift_buf, CL_FALSE, 0, emb_sz, emb_shift, 0, nullptr, nullptr));

    // Upload norm1 gamma/beta to GPU (reuse scale/bias buffers)
    size_t norm1_sz = (size_t)C_in * 4u;
    static cl_mem g_norm1_gamma_buf = nullptr;
    static size_t g_norm1_gamma_sz = 0;
    static cl_mem g_norm1_beta_buf = nullptr;
    static size_t g_norm1_beta_sz = 0;
    ensure_cl_buf(g_norm1_gamma_buf, g_norm1_gamma_sz, norm1_sz, CL_MEM_READ_ONLY);
    ensure_cl_buf(g_norm1_beta_buf, g_norm1_beta_sz, norm1_sz, CL_MEM_READ_ONLY);
    CL_CHECK(clEnqueueWriteBuffer(g_queue, g_norm1_gamma_buf, CL_FALSE, 0, norm1_sz, norm1_gamma, 0, nullptr, nullptr));
    CL_CHECK(clEnqueueWriteBuffer(g_queue, g_norm1_beta_buf, CL_FALSE, 0, norm1_sz, norm1_beta, 0, nullptr, nullptr));

    // 1. Upload input FP32 → GPU via pinned-mapped buffer (avoids unpinned DMA bounce)
    static bool s_frb_profile = (std::getenv("RB_PROFILE") != nullptr);
    auto frb_now = []() -> double {
        auto tp = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(tp.time_since_epoch()).count();
    };
    double t0_frb = s_frb_profile ? frb_now() : 0.0;
    bool prefetch_used = false;
    if (g_prefetch_event && g_prefetch_ptr == in_f32 && g_prefetch_sz == in_f32_sz) {
        // DMA was pre-started; wait for completion (may already be done)
        clWaitForEvents(1, &g_prefetch_event);
        clReleaseEvent(g_prefetch_event);
        g_prefetch_event = nullptr;
        g_prefetch_ptr   = nullptr;
        g_prefetch_sz    = 0;
        prefetch_used = true;
    } else {
        // Blocking upload: DMA from host to GPU VRAM (slow: ~2.76GB/s PCIe BW)
        CL_CHECK(clEnqueueWriteBuffer(g_queue, g_feat_fp32_buf, CL_TRUE,
                                      0, in_f32_sz, in_f32, 0, nullptr, nullptr));
    }
    if (s_frb_profile) { clFinish(g_queue); double tmap = frb_now(); fprintf(stderr, "  [frb_sub] Cin=%d map+memcpy+unmap=%.1fms (prefetch=%d)\n", C_in, (tmap-t0_frb)*1000.0, (int)prefetch_used); t0_frb = tmap; }
    double t1_frb = 0.0, t2_frb = 0.0, t3_frb = 0.0;
    // Neighbor map: re-upload to GPU only when coords changed (nmap identical across all
    // resblocks within an ODE step and across all 25 steps). Saves a blocking 1-2 MB write/call.
    static bool  s_gpu_nmap_valid = false;
    static int   s_gpu_nmap_N     = 0;
    bool nmap_need_upload = (!nmap_host_cached) || (!g_gpu_nmap_valid) || (g_gpu_nmap_N != N);
    if (nmap_need_upload) {
        CL_CHECK(clEnqueueWriteBuffer(g_queue, g_neighbor_map_buf, CL_TRUE, 0, nmap_sz, nmap, 0, nullptr, nullptr));
        g_gpu_nmap_valid = true;
        g_gpu_nmap_N     = N;
    }
    {
        int n_el = N * C_in;
        CL_CHECK(clSetKernelArg(g_float_to_half_kernel, 0, sizeof(cl_mem), &g_feat_fp32_buf));
        CL_CHECK(clSetKernelArg(g_float_to_half_kernel, 1, sizeof(cl_mem), &g_feat_buf[0]));
        CL_CHECK(clSetKernelArg(g_float_to_half_kernel, 2, sizeof(int), &n_el));
        size_t gws = (size_t)((n_el + 255) / 256) * 256;
        size_t lws = 256;
        CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_float_to_half_kernel, 1, nullptr,
                                         &gws, &lws, 0, nullptr, nullptr));
    }
    if (s_frb_profile) { clFinish(g_queue); double tf2h = frb_now(); fprintf(stderr, "  [frb_sub] Cin=%d upload+f2h=%.1fms\n", C_in, (tf2h-t0_frb)*1000.0); t0_frb = tf2h; }

    // 2. Copy identity: g_identity_buf = g_feat_buf[0]
    {
        int total = N * C_in;
        CL_CHECK(clSetKernelArg(g_copy_fp16_kernel, 0, sizeof(cl_mem), &g_feat_buf[0]));
        CL_CHECK(clSetKernelArg(g_copy_fp16_kernel, 1, sizeof(cl_mem), &g_identity_buf));
        CL_CHECK(clSetKernelArg(g_copy_fp16_kernel, 2, sizeof(int), &total));
        size_t gws = (size_t)((total + 255) / 256) * 256;
        size_t lws = 256;
        CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_copy_fp16_kernel, 1, nullptr,
                                         &gws, &lws, 0, nullptr, nullptr));
    }

    // 3. Norm1 + SiLU on GPU: g_feat_buf[0] → g_feat_buf[1]
    {
        float eps = 1e-6f;
        CL_CHECK(clSetKernelArg(g_layernorm_silu_fp16_kernel, 0, sizeof(cl_mem), &g_feat_buf[0]));
        CL_CHECK(clSetKernelArg(g_layernorm_silu_fp16_kernel, 1, sizeof(cl_mem), &g_norm1_gamma_buf));
        CL_CHECK(clSetKernelArg(g_layernorm_silu_fp16_kernel, 2, sizeof(cl_mem), &g_norm1_beta_buf));
        CL_CHECK(clSetKernelArg(g_layernorm_silu_fp16_kernel, 3, sizeof(int), &N));
        CL_CHECK(clSetKernelArg(g_layernorm_silu_fp16_kernel, 4, sizeof(int), &C_in));
        CL_CHECK(clSetKernelArg(g_layernorm_silu_fp16_kernel, 5, sizeof(float), &eps));
        CL_CHECK(clSetKernelArg(g_layernorm_silu_fp16_kernel, 6, sizeof(cl_mem), &g_feat_buf[1]));
        size_t gws = (size_t)((N + 255) / 256) * 256;
        size_t lws = 256;
        CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_layernorm_silu_fp16_kernel, 1, nullptr,
                                         &gws, &lws, 0, nullptr, nullptr));
    }

    // 4. Conv1: g_feat_buf[1] → g_feat_buf[0] (in-place on GPU)
    //    Swap src/dst: input is in buf[1] from norm1, output goes to buf[0]
    //    But gpu_subm_conv3d_inplace expects src_idx and dst_idx
    //    First swap buf[0] and buf[1] semantics by copying buf[1]→buf[0]
    //    Actually simpler: just pass src=1, dst=0
    gpu_subm_conv3d_inplace(conv1_w, conv1_s, conv1_b, N, C_in, C_mid, 1, 0);
    if (s_frb_profile) { clFinish(g_queue); t1_frb = frb_now(); }

    // 5. Norm2_act on GPU: g_feat_buf[0] → g_feat_buf[1]
    //    layernorm(no affine) * (1+scale) + shift → SiLU
    {
        float eps = 1e-6f;
        CL_CHECK(clSetKernelArg(g_layernorm_ss_silu_fp16_kernel, 0, sizeof(cl_mem), &g_feat_buf[0]));
        CL_CHECK(clSetKernelArg(g_layernorm_ss_silu_fp16_kernel, 1, sizeof(cl_mem), &g_emb_scale_buf));
        CL_CHECK(clSetKernelArg(g_layernorm_ss_silu_fp16_kernel, 2, sizeof(cl_mem), &g_emb_shift_buf));
        CL_CHECK(clSetKernelArg(g_layernorm_ss_silu_fp16_kernel, 3, sizeof(int), &N));
        CL_CHECK(clSetKernelArg(g_layernorm_ss_silu_fp16_kernel, 4, sizeof(int), &C_mid));
        CL_CHECK(clSetKernelArg(g_layernorm_ss_silu_fp16_kernel, 5, sizeof(float), &eps));
        CL_CHECK(clSetKernelArg(g_layernorm_ss_silu_fp16_kernel, 6, sizeof(cl_mem), &g_feat_buf[1]));
        size_t gws = (size_t)((N + 255) / 256) * 256;
        size_t lws = 256;
        CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_layernorm_ss_silu_fp16_kernel, 1, nullptr,
                                         &gws, &lws, 0, nullptr, nullptr));
    }

    // 6. Conv2: g_feat_buf[1] → g_feat_buf[0]
    gpu_subm_conv3d_inplace(conv2_w, conv2_s, conv2_b, N, C_mid, C_out, 1, 0);
    // Result in g_feat_buf[0]
    if (s_frb_profile) { clFinish(g_queue); t2_frb = frb_now(); }

    // 7. Skip connection
    if (C_in != C_out && skip_w != nullptr) {
        // Linear projection: g_identity_buf [N,C_in] → g_feat_buf[1] [N,C_out]
        // Skip weights are constant across all ODE steps → convert FP16 + upload ONCE,
        // cached by weight pointer (stable for a given OV compiled model).
        size_t skip_w_sz = (size_t)C_in * C_out * 2u;
        size_t skip_b_sz = (size_t)C_out * 4u;
        struct SkipCache { cl_mem w = nullptr; cl_mem b = nullptr; };
        static std::unordered_map<const float*, SkipCache> g_skip_cache;
        auto it_sk = g_skip_cache.find(skip_w);
        cl_mem g_skip_w_buf, g_skip_b_buf;
        if (it_sk == g_skip_cache.end()) {
            cl_int e;
            SkipCache sc;
            sc.w = clCreateBuffer(g_context, CL_MEM_READ_ONLY, skip_w_sz, nullptr, &e); CL_CHECK(e);
            sc.b = clCreateBuffer(g_context, CL_MEM_READ_ONLY, skip_b_sz, nullptr, &e); CL_CHECK(e);
            size_t n_sw = (size_t)C_in * C_out;
            std::vector<uint16_t> sw_f16(n_sw);
            for (size_t i = 0; i < n_sw; i++) sw_f16[i] = f32_to_f16(skip_w[i]);
            CL_CHECK(clEnqueueWriteBuffer(g_queue, sc.w, CL_FALSE, 0, skip_w_sz, sw_f16.data(), 0, nullptr, nullptr));
            CL_CHECK(clEnqueueWriteBuffer(g_queue, sc.b, CL_TRUE,  0, skip_b_sz, skip_b,        0, nullptr, nullptr));
            g_skip_cache[skip_w] = sc;
            g_skip_w_buf = sc.w; g_skip_b_buf = sc.b;
        } else {
            g_skip_w_buf = it_sk->second.w;
            g_skip_b_buf = it_sk->second.b;
        }

        // linear_fp16: g_identity_buf × skip_w → g_feat_buf[1]
        int has_bias = 1;
        CL_CHECK(clSetKernelArg(g_linear_fp16_kernel, 0, sizeof(cl_mem), &g_identity_buf));
        CL_CHECK(clSetKernelArg(g_linear_fp16_kernel, 1, sizeof(cl_mem), &g_skip_w_buf));
        CL_CHECK(clSetKernelArg(g_linear_fp16_kernel, 2, sizeof(cl_mem), &g_skip_b_buf));
        CL_CHECK(clSetKernelArg(g_linear_fp16_kernel, 3, sizeof(int), &N));
        CL_CHECK(clSetKernelArg(g_linear_fp16_kernel, 4, sizeof(int), &C_in));
        CL_CHECK(clSetKernelArg(g_linear_fp16_kernel, 5, sizeof(int), &C_out));
        CL_CHECK(clSetKernelArg(g_linear_fp16_kernel, 6, sizeof(cl_mem), &g_feat_buf[1]));
        CL_CHECK(clSetKernelArg(g_linear_fp16_kernel, 7, sizeof(int), &has_bias));
        size_t gws[2] = {(size_t)N, (size_t)C_out};
        size_t lws[2] = {std::min((size_t)N, (size_t)64), std::min((size_t)C_out, (size_t)16)};
        CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_linear_fp16_kernel, 2, nullptr,
                                         gws, lws, 0, nullptr, nullptr));

        // 8. Residual add: g_feat_buf[0] += g_feat_buf[1]
        int total = N * C_out;
        CL_CHECK(clSetKernelArg(g_residual_add_fp16_kernel, 0, sizeof(cl_mem), &g_feat_buf[0]));
        CL_CHECK(clSetKernelArg(g_residual_add_fp16_kernel, 1, sizeof(cl_mem), &g_feat_buf[1]));
        CL_CHECK(clSetKernelArg(g_residual_add_fp16_kernel, 2, sizeof(cl_mem), &g_feat_buf[0]));
        CL_CHECK(clSetKernelArg(g_residual_add_fp16_kernel, 3, sizeof(int), &total));
        size_t gws1 = (size_t)((total + 255) / 256) * 256;
        size_t lws1 = 256;
        CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_residual_add_fp16_kernel, 1, nullptr,
                                         &gws1, &lws1, 0, nullptr, nullptr));
    } else {
        // Identity skip: g_feat_buf[0] += g_identity_buf
        int total = N * C_out;
        CL_CHECK(clSetKernelArg(g_residual_add_fp16_kernel, 0, sizeof(cl_mem), &g_feat_buf[0]));
        CL_CHECK(clSetKernelArg(g_residual_add_fp16_kernel, 1, sizeof(cl_mem), &g_identity_buf));
        CL_CHECK(clSetKernelArg(g_residual_add_fp16_kernel, 2, sizeof(cl_mem), &g_feat_buf[0]));
        CL_CHECK(clSetKernelArg(g_residual_add_fp16_kernel, 3, sizeof(int), &total));
        size_t gws = (size_t)((total + 255) / 256) * 256;
        size_t lws = 256;
        CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_residual_add_fp16_kernel, 1, nullptr,
                                         &gws, &lws, 0, nullptr, nullptr));
    }

    // 9. Download: FP16 → FP32 → CPU
    {
        int n_el = N * C_out;
        CL_CHECK(clSetKernelArg(g_half_to_float_kernel, 0, sizeof(cl_mem), &g_feat_buf[0]));
        CL_CHECK(clSetKernelArg(g_half_to_float_kernel, 1, sizeof(cl_mem), &g_out_f32_buf));
        CL_CHECK(clSetKernelArg(g_half_to_float_kernel, 2, sizeof(int), &n_el));
        size_t gws = (size_t)((n_el + 255) / 256) * 256;
        size_t lws = 256;
        CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_half_to_float_kernel, 1, nullptr,
                                         &gws, &lws, 0, nullptr, nullptr));
    }
    CL_CHECK(clEnqueueReadBuffer(g_queue, g_out_f32_buf, CL_TRUE, 0,
                                  out_f32_sz, out_f32, 0, nullptr, nullptr));
    if (s_frb_profile) {
        double t_end = frb_now();
        t3_frb = t_end;
        fprintf(stderr, "  [frb] Cin=%d N=%d: upload→conv1=%.1fms  conv2=%.1fms  skip+dl=%.1fms  total=%.1fms\n",
                C_in, N,
                (t1_frb - t0_frb)*1000.0,
                (t2_frb - t1_frb)*1000.0,
                (t3_frb - t2_frb)*1000.0,
                (t3_frb - t0_frb)*1000.0);
        (void)t3_frb;
    }
}

// ============================================================
// FP32 Mesh Upsample (SparseMeshUpsample op)
// Self-contained, accuracy-critical path feeding FlexiCubes.
// Convention matches CPU reference (mesh_extract_ref.py) exactly.
// ============================================================

// Build neighbor map for mesh convs. coords: [N,4] i32 (b,z,y,x), single batch.
// nmap[p*27+k] = index of voxel at (z+kz-1, y+ky-1, x+kx-1), k=kz*9+ky*3+kx; -1 if absent.
static void build_mesh_neighbor_map(const int32_t* coords, int N, int32_t* nmap) {    std::unordered_map<int64_t, int32_t> h;
    h.reserve((size_t)N * 2);
    auto key = [](int z, int y, int x) -> int64_t {
        return (((int64_t)(z + 1) * 4096) + (y + 1)) * 4096 + (x + 1);
    };
    for (int i = 0; i < N; i++) {
        int z = coords[i*4+1], y = coords[i*4+2], x = coords[i*4+3];
        h[key(z, y, x)] = i;
    }
    for (int i = 0; i < N; i++) {
        int z = coords[i*4+1], y = coords[i*4+2], x = coords[i*4+3];
        for (int kz = 0; kz < 3; kz++)
            for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++) {
                    int k = kz*9 + ky*3 + kx;
                    auto it = h.find(key(z + kz - 1, y + ky - 1, x + kx - 1));
                    nmap[(size_t)i*27 + k] = (it == h.end()) ? -1 : it->second;
                }
    }
}

// Subdivide coords: each voxel -> 8 children. coords_out[n*8+child] = coords[n]*2 + offset.
// offsets lexicographic {0,1}^3 over (z,y,x) (matches ref _SUBDIV_OFFSETS).
static void subdivide_coords(const int32_t* coords, int N, int32_t* coords_out) {
    static const int off[8][3] = {
        {0,0,0},{0,0,1},{0,1,0},{0,1,1},{1,0,0},{1,0,1},{1,1,0},{1,1,1}
    };
    for (int n = 0; n < N; n++) {
        int b = coords[n*4+0];
        int z = coords[n*4+1], y = coords[n*4+2], x = coords[n*4+3];
        for (int c = 0; c < 8; c++) {
            int p = n*8 + c;
            coords_out[p*4+0] = b;
            coords_out[p*4+1] = z*2 + off[c][0];
            coords_out[p*4+2] = y*2 + off[c][1];
            coords_out[p*4+3] = x*2 + off[c][2];
        }
    }
}

static size_t g_mesh_live_bytes = 0;

static cl_mem mesh_mk_buf(size_t bytes) {
    cl_int e; cl_mem m = clCreateBuffer(g_context, CL_MEM_READ_WRITE, bytes, nullptr, &e);
    if (e != CL_SUCCESS) {
        cl_ulong gmem = 0, amax = 0;
        clGetDeviceInfo(g_device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, nullptr);
        clGetDeviceInfo(g_device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(amax), &amax, nullptr);
        fprintf(stderr, "[mesh] clCreateBuffer(%.1f MB) failed err=%d | live=%.1f MB "
                        "device_global=%.1f MB max_alloc=%.1f MB\n",
                bytes / 1048576.0, (int)e, g_mesh_live_bytes / 1048576.0,
                gmem / 1048576.0, amax / 1048576.0);
        fflush(stderr);
    }
    CL_CHECK(e);
    g_mesh_live_bytes += bytes;
    if (getenv("SAM3D_MESH_MEM"))
        fprintf(stderr, "[mesh_mem] +%.1f MB -> live %.1f MB\n",
                bytes / 1048576.0, g_mesh_live_bytes / 1048576.0);
    return m;
}
static void mesh_free(cl_mem m, size_t bytes) {
    clReleaseMemObject(m);
    g_mesh_live_bytes -= bytes;
}
static cl_mem mesh_mk_ro(const void* host, size_t bytes) {
    // Allocate device-only then upload with an explicit blocking write.  Using
    // CL_MEM_COPY_HOST_PTR here was observed to intermittently poison the Intel
    // NEO context (deferred CL_OUT_OF_RESOURCES surfacing on a later enqueue).
    cl_int e; cl_mem m = clCreateBuffer(g_context, CL_MEM_READ_ONLY, bytes, nullptr, &e);
    CL_CHECK(e);
    CL_CHECK(clEnqueueWriteBuffer(g_queue, m, CL_TRUE, 0, bytes, host, 0, nullptr, nullptr));
    return m;
}

// Tile sizes must match MESH_TP / MESH_TCO in sparse_conv_3d.cl.
static const int MESH_TP  = 4;
static const int MESH_TCO = 8;

// Transpose conv weights [Cout, 27, Cin] -> [27, Cin, Cout] so the tiled
// kernel's MESH_TCO output channels land contiguously.
static void mesh_transpose_weight(const float* w, int Cin, int Cout,
                                  std::vector<float>& out) {
    out.resize((size_t)27 * Cin * Cout);
    for (int co = 0; co < Cout; co++)
        for (int k = 0; k < 27; k++) {
            const float* src = w + ((size_t)co * 27 + k) * Cin;
            float* dst = out.data() + (size_t)k * Cin * Cout + co;
            for (int ci = 0; ci < Cin; ci++) dst[(size_t)ci * Cout] = src[ci];
        }
}

// Runs one submanifold conv3d.  Takes the weights as host pointers because the
// scalar and tiled kernels want different layouts; the upload happens here.
// row_shift=3 makes the kernel gather from the coarse [N/8, Cin] buffer instead
// of a materialised subdivided copy.
static void mesh_run_conv3d(cl_mem feat_in, const float* w_host, const float* b_host,
                            cl_mem nmap, int N, int Cin, int Cout, cl_mem feat_out,
                            int row_shift = 0) {
    // Chunk the voxel dimension so no single GPU submission runs long enough to
    // trip the kernel-execution watchdog (naive fp32 conv is compute-heavy).
    static const size_t CHUNK = [] {
        const char* e = getenv("SAM3D_MESH_CONV_CHUNK");
        return e ? (size_t)std::max(1, atoi(e)) : (size_t)1024;
    }();

    cl_mem bias = mesh_mk_ro(b_host, (size_t)Cout * 4);

    // Tiled path needs the transposed weight layout and Cout % MESH_TCO == 0.
    const bool tiled = (Cout % MESH_TCO == 0)
                       && getenv("SAM3D_MESH_CONV_SCALAR") == nullptr;
    if (tiled) {
        std::vector<float> wt;
        mesh_transpose_weight(w_host, Cin, Cout, wt);
        cl_mem weight = mesh_mk_ro(wt.data(), wt.size() * 4);

        cl_kernel k = g_subm_conv3d_fp32_tiled_kernel;
        CL_CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &feat_in));
        CL_CHECK(clSetKernelArg(k, 1, sizeof(cl_mem), &weight));
        CL_CHECK(clSetKernelArg(k, 2, sizeof(cl_mem), &bias));
        CL_CHECK(clSetKernelArg(k, 3, sizeof(cl_mem), &nmap));
        CL_CHECK(clSetKernelArg(k, 4, sizeof(int), &N));
        CL_CHECK(clSetKernelArg(k, 5, sizeof(int), &Cin));
        CL_CHECK(clSetKernelArg(k, 6, sizeof(int), &Cout));
        CL_CHECK(clSetKernelArg(k, 9, sizeof(int), &row_shift));
        CL_CHECK(clSetKernelArg(k, 10, sizeof(cl_mem), &feat_out));
        for (size_t start = 0; start < (size_t)N; start += CHUNK) {
            int cnt = (int)std::min(CHUNK, (size_t)N - start);
            int p_off = (int)start;
            CL_CHECK(clSetKernelArg(k, 7, sizeof(int), &p_off));
            CL_CHECK(clSetKernelArg(k, 8, sizeof(int), &cnt));
            size_t gws[2] = {(size_t)(Cout / MESH_TCO),
                             (size_t)((cnt + MESH_TP - 1) / MESH_TP)};
            CL_CHECK(clEnqueueNDRangeKernel(g_queue, k, 2, nullptr, gws, nullptr,
                                            0, nullptr, nullptr));
            CL_CHECK(clFinish(g_queue));
        }
        clReleaseMemObject(weight);
        clReleaseMemObject(bias);
        return;
    }

    cl_mem weight = mesh_mk_ro(w_host, (size_t)Cout * 27 * Cin * 4);
    cl_kernel k = g_subm_conv3d_fp32_kernel;
    CL_CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &feat_in));
    CL_CHECK(clSetKernelArg(k, 1, sizeof(cl_mem), &weight));
    CL_CHECK(clSetKernelArg(k, 2, sizeof(cl_mem), &bias));
    CL_CHECK(clSetKernelArg(k, 3, sizeof(cl_mem), &nmap));
    CL_CHECK(clSetKernelArg(k, 4, sizeof(int), &N));
    CL_CHECK(clSetKernelArg(k, 5, sizeof(int), &Cin));
    CL_CHECK(clSetKernelArg(k, 6, sizeof(int), &Cout));
    CL_CHECK(clSetKernelArg(k, 8, sizeof(int), &row_shift));
    CL_CHECK(clSetKernelArg(k, 9, sizeof(cl_mem), &feat_out));
    for (size_t start = 0; start < (size_t)N; start += CHUNK) {
        size_t cnt = std::min(CHUNK, (size_t)N - start);
        int p_off = (int)start;
        CL_CHECK(clSetKernelArg(k, 7, sizeof(int), &p_off));
        size_t gws[2] = {cnt, (size_t)Cout};
        cl_int ee = clEnqueueNDRangeKernel(g_queue, k, 2, nullptr, gws, nullptr, 0, nullptr, nullptr);
        CL_CHECK(ee);
        CL_CHECK(clFinish(g_queue));
    }
    clReleaseMemObject(weight);
    clReleaseMemObject(bias);
}

static void mesh_run_linear(cl_mem feat_in, cl_mem weight, cl_mem bias,
                            int N, int Cin, int Cout, cl_mem feat_out,
                            int row_shift = 0, int accumulate = 0) {
    cl_kernel k = g_linear_fp32_kernel;
    CL_CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &feat_in));
    CL_CHECK(clSetKernelArg(k, 1, sizeof(cl_mem), &weight));
    CL_CHECK(clSetKernelArg(k, 2, sizeof(cl_mem), &bias));
    CL_CHECK(clSetKernelArg(k, 3, sizeof(int), &N));
    CL_CHECK(clSetKernelArg(k, 4, sizeof(int), &Cin));
    CL_CHECK(clSetKernelArg(k, 5, sizeof(int), &Cout));
    CL_CHECK(clSetKernelArg(k, 6, sizeof(cl_mem), &feat_out));
    CL_CHECK(clSetKernelArg(k, 9, sizeof(int), &row_shift));
    CL_CHECK(clSetKernelArg(k, 10, sizeof(int), &accumulate));
    // Chunk the voxel dimension so no single GPU submission runs long enough to
    // trip the kernel-execution watchdog (mirrors mesh_run_conv3d).  A single
    // unchunked enqueue over N*Cout work items (N up to ~580k) trips CL -5
    // (CL_OUT_OF_RESOURCES) on the Intel iGPU watchdog.
    const size_t CHUNK = 4096;   // voxels per submission
    const size_t lws = 256;
    for (size_t start = 0; start < (size_t)N; start += CHUNK) {
        int cnt = (int)std::min(CHUNK, (size_t)N - start);
        int p_off = (int)start;
        CL_CHECK(clSetKernelArg(k, 7, sizeof(int), &p_off));
        CL_CHECK(clSetKernelArg(k, 8, sizeof(int), &cnt));
        size_t gws = (size_t)cnt * Cout;
        size_t gws_r = ((gws + lws - 1) / lws) * lws;
        CL_CHECK(clEnqueueNDRangeKernel(g_queue, k, 1, nullptr, &gws_r, &lws, 0, nullptr, nullptr));
        CL_CHECK(clFinish(g_queue));
    }
}

// NOTE: the standalone subdivide and elementwise-add passes that used to sit
// here are gone.  The subdivide gather is folded into the conv/linear kernels
// via their row_shift argument, and the skip-connection add is folded into the
// linear kernel via its accumulate argument.  This removed ~2.3 GB of device
// buffers on the second block, which is what was intermittently tripping
// CL_OUT_OF_RESOURCES at enqueue time.  subdivide_tile_fp32 / add_fp32 are now
// unreferenced but are kept compiled as a fallback for A/B debugging.

// GroupNorm32 (over N) + optional SiLU, FP32. gamma/beta host pointers [C].
static void mesh_run_gn_silu(cl_mem feat_in, int N, int C,
                             const float* gamma, const float* beta, int silu,
                             cl_mem feat_out) {
    const int num_groups = 32;
    const float eps = 1e-5f;
    int cg = C / num_groups;

    // Parallel reduction over N: split into P partitions so no single work-item
    // loops over all N (a C-thread serial reduce trips the GPU watchdog for large
    // N). Each (c,p) sums a strided slice; the P partials are summed on the host.
    const int P = 256;
    cl_mem part_sum   = mesh_mk_buf((size_t)C * P * 4);
    cl_mem part_sumsq = mesh_mk_buf((size_t)C * P * 4);

    // Pass 1: partial sum + sumsq
    {
        cl_kernel k = g_gn_partial_fp32_kernel;
        CL_CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &feat_in));
        CL_CHECK(clSetKernelArg(k, 1, sizeof(int), &N));
        CL_CHECK(clSetKernelArg(k, 2, sizeof(int), &C));
        CL_CHECK(clSetKernelArg(k, 3, sizeof(int), &P));
        CL_CHECK(clSetKernelArg(k, 4, sizeof(cl_mem), &part_sum));
        CL_CHECK(clSetKernelArg(k, 5, sizeof(cl_mem), &part_sumsq));
        size_t gws[2] = {(size_t)C, (size_t)P};
        CL_CHECK(clEnqueueNDRangeKernel(g_queue, k, 2, nullptr, gws, nullptr, 0, nullptr, nullptr));
    }
    std::vector<float> psum((size_t)C * P);
    CL_CHECK(clEnqueueReadBuffer(g_queue, part_sum, CL_TRUE, 0, (size_t)C*P*4, psum.data(), 0, nullptr, nullptr));

    // Per-channel sum -> per-group mean (expanded per-channel)
    std::vector<float> sum(C, 0.0f);
    for (int c = 0; c < C; c++) {
        double s = 0.0;
        for (int p = 0; p < P; p++) s += psum[(size_t)c*P + p];
        sum[c] = (float)s;
    }
    std::vector<float> mean(C), inv_std(C);
    double cnt = (double)cg * (double)N;
    for (int g = 0; g < num_groups; g++) {
        double gs = 0.0;
        for (int j = 0; j < cg; j++) gs += sum[g*cg+j];
        double m = gs / cnt;
        for (int j = 0; j < cg; j++) mean[g*cg+j] = (float)m;
    }
    cl_mem mean_buf = mesh_mk_ro(mean.data(), (size_t)C*4);

    // Pass 2: partial sum of squared deviations from mean (stable variance)
    {
        cl_kernel k = g_gn_partial_sqdev_fp32_kernel;
        CL_CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &feat_in));
        CL_CHECK(clSetKernelArg(k, 1, sizeof(cl_mem), &mean_buf));
        CL_CHECK(clSetKernelArg(k, 2, sizeof(int), &N));
        CL_CHECK(clSetKernelArg(k, 3, sizeof(int), &C));
        CL_CHECK(clSetKernelArg(k, 4, sizeof(int), &P));
        CL_CHECK(clSetKernelArg(k, 5, sizeof(cl_mem), &part_sumsq));
        size_t gws[2] = {(size_t)C, (size_t)P};
        CL_CHECK(clEnqueueNDRangeKernel(g_queue, k, 2, nullptr, gws, nullptr, 0, nullptr, nullptr));
    }
    std::vector<float> psq((size_t)C * P);
    CL_CHECK(clEnqueueReadBuffer(g_queue, part_sumsq, CL_TRUE, 0, (size_t)C*P*4, psq.data(), 0, nullptr, nullptr));

    std::vector<float> sqdev(C, 0.0f);
    for (int c = 0; c < C; c++) {
        double s = 0.0;
        for (int p = 0; p < P; p++) s += psq[(size_t)c*P + p];
        sqdev[c] = (float)s;
    }
    for (int g = 0; g < num_groups; g++) {
        double gsq = 0.0;
        for (int j = 0; j < cg; j++) gsq += sqdev[g*cg+j];
        double var = gsq / cnt;
        double istd = 1.0 / std::sqrt(var + eps);
        for (int j = 0; j < cg; j++) inv_std[g*cg+j] = (float)istd;
    }

    cl_mem istd_buf = mesh_mk_ro(inv_std.data(), (size_t)C*4);
    cl_mem gam_buf  = mesh_mk_ro(gamma,          (size_t)C*4);
    cl_mem beta_buf = mesh_mk_ro(beta,           (size_t)C*4);
    {
        cl_kernel k = g_gn_affine_silu_fp32_kernel;
        CL_CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &feat_in));
        CL_CHECK(clSetKernelArg(k, 1, sizeof(cl_mem), &mean_buf));
        CL_CHECK(clSetKernelArg(k, 2, sizeof(cl_mem), &istd_buf));
        CL_CHECK(clSetKernelArg(k, 3, sizeof(cl_mem), &gam_buf));
        CL_CHECK(clSetKernelArg(k, 4, sizeof(cl_mem), &beta_buf));
        CL_CHECK(clSetKernelArg(k, 5, sizeof(int), &N));
        CL_CHECK(clSetKernelArg(k, 6, sizeof(int), &C));
        CL_CHECK(clSetKernelArg(k, 7, sizeof(int), &silu));
        CL_CHECK(clSetKernelArg(k, 8, sizeof(cl_mem), &feat_out));
        size_t gws[2] = {(size_t)N, (size_t)C};
        CL_CHECK(clEnqueueNDRangeKernel(g_queue, k, 2, nullptr, gws, nullptr, 0, nullptr, nullptr));
        CL_CHECK(clFinish(g_queue));
    }

    clReleaseMemObject(part_sum);  clReleaseMemObject(part_sumsq);
    clReleaseMemObject(mean_buf);  clReleaseMemObject(istd_buf);
    clReleaseMemObject(gam_buf);   clReleaseMemObject(beta_buf);
}

// Weight pointers for one SparseSubdivideBlock3d.
struct MeshBlockW {
    const float* act_gamma; const float* act_beta;       // act_layers.0 (GN, Cin)
    const float* conv0_w;   const float* conv0_b;        // out_layers.0.conv (Cmid,27,Cin)
    const float* gn1_gamma; const float* gn1_beta;       // out_layers.1 (GN, Cmid)
    const float* conv3_w;   const float* conv3_b;        // out_layers.3.conv (Cout,27,Cmid)
    const float* skip_w;    const float* skip_b;         // skip_connection.conv (Cout,Cin) k1
    int Cin, Cmid, Cout;
};

// Process one subdivide block. feat_in_dev: [N,Cin] fp32 device. coords_in: [N,4] host.
// Returns device buffer [Nf=8N, Cout] and writes coords_out (host, [8N,4]). Nf via *Nf_out.
static cl_mem mesh_process_block(cl_mem feat_in_dev, const int32_t* coords_in, int N,
                                 const MeshBlockW& w,
                                 std::vector<int32_t>& coords_out, int* Nf_out) {
    int Cin = w.Cin, Cmid = w.Cmid, Cout = w.Cout;
    int Nf = N * 8;
    *Nf_out = Nf;

    const bool prof = (getenv("SAM3D_MESH_PROFILE") != nullptr);
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto t0 = now();

    // 1. act: GN+SiLU on feat_in (N,Cin) -> h (N,Cin)
    cl_mem h = mesh_mk_buf((size_t)N * Cin * 4);
    mesh_run_gn_silu(feat_in_dev, N, Cin, w.act_gamma, w.act_beta, /*silu=*/1, h);
    auto t1 = now();

    // 2. subdivide coords + build fine neighbor map
    coords_out.resize((size_t)Nf * 4);
    subdivide_coords(coords_in, N, coords_out.data());
    auto t2 = now();
    std::vector<int32_t> nmap((size_t)Nf * 27);
    build_mesh_neighbor_map(coords_out.data(), Nf, nmap.data());
    auto t3 = now();
    cl_mem nmap_dev = mesh_mk_ro(nmap.data(), (size_t)Nf * 27 * 4);
    auto t4 = now();

    // 3. conv0 reads the COARSE h with row_shift=3, which is exactly what a
    // materialised h_fine[j] == h[j>>3] would have contained.  Skipping the
    // materialisation removes a 1.6 GB device buffer (block 1) plus the
    // bandwidth to write and re-read it.
    auto t5 = now();

    // 4. conv0: h (subdivided on the fly) -> (Nf,Cmid)
    const size_t mid_bytes = (size_t)Nf * Cmid * 4;
    cl_mem hc = mesh_mk_buf(mid_bytes);
    mesh_run_conv3d(h, w.conv0_w, w.conv0_b, nmap_dev, Nf, Cin, Cmid, hc, /*row_shift=*/3);
    mesh_free(h, (size_t)N * Cin * 4);
    auto t6 = now();

    // 5. gn1 + SiLU (Nf,Cmid)
    cl_mem hn = mesh_mk_buf(mid_bytes);
    mesh_run_gn_silu(hc, Nf, Cmid, w.gn1_gamma, w.gn1_beta, /*silu=*/1, hn);
    mesh_free(hc, mid_bytes);
    auto t7 = now();

    // 6. conv3: (Nf,Cmid) -> (Nf,Cout)
    const size_t out_bytes = (size_t)Nf * Cout * 4;
    cl_mem hout = mesh_mk_buf(out_bytes);
    mesh_run_conv3d(hn, w.conv3_w, w.conv3_b, nmap_dev, Nf, Cmid, Cout, hout);
    mesh_free(hn, mid_bytes);
    clReleaseMemObject(nmap_dev);
    auto t8 = now();

    // 7. skip: linear k1 over the subdivided input, accumulated straight onto
    // the conv3 output.  Reading feat_in with row_shift=3 avoids materialising
    // x_fine, and accumulating in place avoids both a separate skip buffer and
    // a separate output buffer -- together ~2.3 GB of device memory for block 1,
    // which is what pushed the op into CL_OUT_OF_RESOURCES when the OV GPU
    // plugin was simultaneously holding the SLAT models.
    cl_mem skip_w_dev = mesh_mk_ro(w.skip_w, (size_t)Cout * Cin * 4);
    cl_mem skip_b_dev = mesh_mk_ro(w.skip_b, (size_t)Cout * 4);
    mesh_run_linear(feat_in_dev, skip_w_dev, skip_b_dev, Nf, Cin, Cout, hout,
                    /*row_shift=*/3, /*accumulate=*/1);
    clReleaseMemObject(skip_w_dev); clReleaseMemObject(skip_b_dev);
    cl_mem out = hout;
    auto t9 = now();

    if (prof) {
        fprintf(stderr,
            "[mesh_prof] block N=%d->Nf=%d Cin=%d Cmid=%d Cout=%d | gn_silu=%.1f "
            "subdiv_coords=%.1f BUILD_NMAP=%.1f nmap_up=%.1f "
            "conv0=%.1f gn1=%.1f conv3=%.1f skip+add=%.1f | total=%.1f ms\n",
            N, Nf, Cin, Cmid, Cout, ms(t0,t1), ms(t1,t2), ms(t2,t3), ms(t3,t4),
            ms(t5,t6), ms(t6,t7), ms(t7,t8), ms(t8,t9), ms(t0,t9));
        fflush(stderr);
    }

    return out;
}

// Full mesh upsample: mesh_hidden_feats (N,768) + coords (N,4) -> (Nf,101) + coords (Nf,4).
// dims = {Cin0,Cmid0,Cout0, Cin1,Cmid1,Cout1, out_in, out_out}
// Returns Nf. out_feats/out_coords must be preallocated to >= Nf.
static int gpu_mesh_upsample(const float* in_feats, const int32_t* in_coords, int N,
                             const float* params, const int64_t* dims,
                             float* out_feats, int32_t* out_coords) {
    std::call_once(g_init_flag, init_opencl);

    int Cin0 = (int)dims[0], Cmid0 = (int)dims[1], Cout0 = (int)dims[2];
    int Cin1 = (int)dims[3], Cmid1 = (int)dims[4], Cout1 = (int)dims[5];
    int olin = (int)dims[6], olout = (int)dims[7];

    // Parse packed params into per-tensor pointers (see export packing order)
    size_t off = 0;
    auto take = [&](size_t n) -> const float* { const float* p = params + off; off += n; return p; };

    MeshBlockW b0;
    b0.Cin = Cin0; b0.Cmid = Cmid0; b0.Cout = Cout0;
    b0.act_gamma = take(Cin0);              b0.act_beta = take(Cin0);
    b0.conv0_w   = take((size_t)Cmid0*27*Cin0); b0.conv0_b = take(Cmid0);
    b0.gn1_gamma = take(Cmid0);             b0.gn1_beta = take(Cmid0);
    b0.conv3_w   = take((size_t)Cout0*27*Cmid0); b0.conv3_b = take(Cout0);
    b0.skip_w    = take((size_t)Cout0*Cin0);    b0.skip_b  = take(Cout0);

    MeshBlockW b1;
    b1.Cin = Cin1; b1.Cmid = Cmid1; b1.Cout = Cout1;
    b1.act_gamma = take(Cin1);              b1.act_beta = take(Cin1);
    b1.conv0_w   = take((size_t)Cmid1*27*Cin1); b1.conv0_b = take(Cmid1);
    b1.gn1_gamma = take(Cmid1);             b1.gn1_beta = take(Cmid1);
    b1.conv3_w   = take((size_t)Cout1*27*Cmid1); b1.conv3_b = take(Cout1);
    b1.skip_w    = take((size_t)Cout1*Cin1);    b1.skip_b  = take(Cout1);

    const float* outw = take((size_t)olout * olin);
    const float* outb = take(olout);

    // Upload input features
    cl_mem feat0 = mesh_mk_ro(in_feats, (size_t)N * Cin0 * 4);

    // Block 0
    std::vector<int32_t> coords1;
    int N1 = 0;
    cl_mem feat1 = mesh_process_block(feat0, in_coords, N, b0, coords1, &N1);
    clReleaseMemObject(feat0);

    // Block 1
    std::vector<int32_t> coords2;
    int N2 = 0;
    cl_mem feat2 = mesh_process_block(feat1, coords1.data(), N1, b1, coords2, &N2);
    clReleaseMemObject(feat1);

    // out_layer: linear Cout1 -> olout
    cl_mem outw_dev = mesh_mk_ro(outw, (size_t)olout * olin * 4);
    cl_mem outb_dev = mesh_mk_ro(outb, (size_t)olout * 4);
    cl_mem out_dev = mesh_mk_buf((size_t)N2 * olout * 4);
    mesh_run_linear(feat2, outw_dev, outb_dev, N2, olin, olout, out_dev);
    clReleaseMemObject(feat2);
    clReleaseMemObject(outw_dev); clReleaseMemObject(outb_dev);

    // Download
    CL_CHECK(clEnqueueReadBuffer(g_queue, out_dev, CL_TRUE, 0,
                                  (size_t)N2 * olout * 4, out_feats, 0, nullptr, nullptr));
    clReleaseMemObject(out_dev);
    std::memcpy(out_coords, coords2.data(), (size_t)N2 * 4 * sizeof(int32_t));

    return N2;
}

// ============================================================
// OV Operation Interface
// ============================================================

SparseConv3dOp::SparseConv3dOp(
    const ov::Output<ov::Node>& features,
    const ov::Output<ov::Node>& coords,
    const ov::Output<ov::Node>& num_voxels,
    const ov::Output<ov::Node>& params)
    : Op({features, coords, num_voxels, params})
{
    constructor_validate_and_infer_types();
}

SparseConv3dOp::SparseConv3dOp(const ov::OutputVector& args)
    : Op(args)
{
    constructor_validate_and_infer_types();
}

void SparseConv3dOp::validate_and_infer_types() {
    // Use actual N from features input if statically known (after reshape);
    // fall back to dynamic if unknown.
    auto features_pshape = get_input_partial_shape(0);
    if (features_pshape.rank().is_static() && features_pshape[0].is_static()) {
        set_output_type(0, ov::element::f32,
            ov::PartialShape{features_pshape[0].get_length(), (int64_t)m_out_channels});
    } else {
        set_output_type(0, ov::element::f32,
            ov::PartialShape{ov::Dimension::dynamic(), (int64_t)m_out_channels});
    }
}

std::shared_ptr<ov::Node> SparseConv3dOp::clone_with_new_inputs(
    const ov::OutputVector& new_args) const
{
    auto op = std::make_shared<SparseConv3dOp>(new_args);
    op->m_max_voxels = m_max_voxels;
    op->m_in_channels = m_in_channels;
    op->m_out_channels = m_out_channels;
    op->m_num_layers = m_num_layers;
    op->m_spatial_resolution = m_spatial_resolution;
    op->m_layer_defs = m_layer_defs;
    return op;
}

bool SparseConv3dOp::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("max_voxels", m_max_voxels);
    visitor.on_attribute("in_channels", m_in_channels);
    visitor.on_attribute("out_channels", m_out_channels);
    visitor.on_attribute("num_layers", m_num_layers);
    visitor.on_attribute("spatial_resolution", m_spatial_resolution);
    visitor.on_attribute("layer_defs", m_layer_defs);
    return true;
}

bool SparseConv3dOp::has_evaluate() const { return true; }

bool SparseConv3dOp::evaluate(
    ov::TensorVector& outputs,
    const ov::TensorVector& inputs) const
{
    const float*   features_ptr   = inputs[0].data<float>();
    const int32_t* coords_ptr     = inputs[1].data<int32_t>();
    const int32_t* num_voxels_ptr = inputs[2].data<int32_t>();
    const float*   params_ptr     = inputs[3].data<float>();
    // Optional inputs for fused resblock (emb_scale, emb_shift from timestep embedding)
    const float*   emb_scale_ptr  = (inputs.size() > 4) ? inputs[4].data<float>() : nullptr;
    const float*   emb_shift_ptr  = (inputs.size() > 5) ? inputs[5].data<float>() : nullptr;

    int N = num_voxels_ptr[0];

    // If N == 0 (e.g. compile-time constant folding with dummy inputs),
    // return a zero output without touching OpenCL/GPU state.
    if (N <= 0) {
        auto& out = outputs[0];
        std::memset(out.data(), 0, out.get_byte_size());
        return true;
    }

    // Initialize OpenCL on first real call (thread-safe)
    std::call_once(g_init_flag, init_opencl);

    if (N > m_max_voxels) N = (int)m_max_voxels;

    int C_in  = (int)m_in_channels;
    int C_out = (int)m_out_channels;
    int num_layers = (int)m_num_layers;

    // Parse layer definitions: [type, cin, cout, nk, type, cin, cout, nk, ...]
    struct LayerDef {
        int type, cin, cout, nk;
    };
    std::vector<LayerDef> layers(num_layers);
    for (int i = 0; i < num_layers; i++) {
        layers[i].type = (int)m_layer_defs[i * 4 + 0];
        layers[i].cin  = (int)m_layer_defs[i * 4 + 1];
        layers[i].cout = (int)m_layer_defs[i * 4 + 2];
        layers[i].nk   = (int)m_layer_defs[i * 4 + 3];
    }

    // Compute weight offsets for each layer
    // Weight packing: [all_weights | all_scales | all_biases]
    size_t total_w = 0, total_s = 0;
    std::vector<size_t> w_offsets(num_layers), s_offsets(num_layers), b_offsets(num_layers);
    for (int i = 0; i < num_layers; i++) {
        w_offsets[i] = total_w;
        total_w += (size_t)layers[i].nk * layers[i].cin * layers[i].cout;
        s_offsets[i] = total_s;
        total_s += layers[i].cout;
    }
    size_t total_b = total_s;  // biases have same count as scales

    const float* all_weights = params_ptr;
    const float* all_scales  = params_ptr + total_w;
    const float* all_biases  = params_ptr + total_w + total_s;

    // Allocate working buffers — use pre-allocated static buffers to avoid
    // per-call heap allocations.  Sized to actual N, not the (large) max_voxels cap.
    int max_ch = 0;
    for (auto& l : layers) {
        max_ch = std::max(max_ch, std::max(l.cin, l.cout));
    }

    // Pre-allocated static buffers — resize only when needed (amortized O(1))
    size_t feat_size = (size_t)N * max_ch;
    if (g_static_buf_a.size() < feat_size) {
        g_static_buf_a.resize(feat_size);
        g_static_buf_b.resize(feat_size);
        g_static_buf_identity.resize(feat_size);
    }
    // NOTE: Do NOT zero-fill buf_a/buf_b/buf_identity here.
    // buf_a: immediately overwritten by the input copy below.
    // buf_b: fully overwritten by the conv GPU output (only N×C_out elements are read back).
    //        Any padding entries (max_ch > C_out) are never read — they can be garbage.
    // buf_identity: never actually read in the current layer implementations.
    float* buf_a_ptr = g_static_buf_a.data();
    float* buf_b_ptr = g_static_buf_b.data();

    // ── Neighbor map: use cache if coordinates haven't changed ──
    bool nmap_from_cache = coords_match(coords_ptr, N);
    int32_t* nmap_ptr;
    if (nmap_from_cache) {
        nmap_ptr = g_cached_nmap.data();
    } else {
        size_t nmap_size = (size_t)N * 27;
        if (g_cached_nmap.size() < nmap_size)
            g_cached_nmap.resize(nmap_size);
        std::fill_n(g_cached_nmap.begin(), nmap_size, -1);
        nmap_ptr = g_cached_nmap.data();
        // Cache the coordinates for future matching
        g_cached_coords.resize((size_t)N * 4);
        std::memcpy(g_cached_coords.data(), coords_ptr, (size_t)N * 4 * sizeof(int32_t));
        g_cached_nmap_N = N;
    }

    // Copy input features to buf_a.
    // When max_ch == C_in (common case: single-layer conv where the first layer has
    // the largest channel count), use a single flat memcpy for ~10% better throughput
    // vs a loop of per-row memcpys.
    if (max_ch == C_in) {
        std::memcpy(buf_a_ptr, features_ptr, (size_t)N * C_in * sizeof(float));
    } else {
        for (int i = 0; i < N; i++) {
            std::memcpy(&buf_a_ptr[(size_t)i * max_ch],
                       &features_ptr[(size_t)i * C_in],
                       C_in * sizeof(float));
        }
    }

    // Current coordinates (may change with strided conv / downsample)
    std::vector<int32_t> cur_coords((size_t)N * 4);
    std::memcpy(cur_coords.data(), coords_ptr, (size_t)N * 4 * sizeof(int32_t));
    int cur_N = N;
    int cur_C = C_in;

    // Track downsampled intermediate states for upsample
    struct SpatialCache {
        std::vector<int32_t> coords;
        int N;
        std::vector<int32_t> upsample_idx;  // [N_fine] → coarse idx
    };
    std::vector<SpatialCache> ds_cache;

    // Build initial neighbor map (only if not cached)
    if (!nmap_from_cache) {
        build_subm_neighbor_map(cur_coords.data(), cur_N, nmap_ptr);
    }

    // ── Detect fused resblock pattern: 5 layers = LAYERNORM_SILU + SUBM_CONV + NORM2_ACT + SUBM_CONV + RESIDUAL_ADD ──
    // If detected, run entirely on GPU via gpu_fused_resblock() — eliminates 5 PCIe round-trips.
    if (num_layers == 5 &&
        layers[0].type == LAYER_LAYERNORM_SILU &&
        layers[1].type == LAYER_SUBM_CONV &&
        layers[2].type == LAYER_NORM2_ACT &&
        layers[3].type == LAYER_SUBM_CONV &&
        layers[4].type == LAYER_RESIDUAL_ADD)
    {
        int C_mid = layers[1].cout;  // conv1 output channels
        // Pointers into packed params:
        const float* norm1_gamma = all_scales + s_offsets[0];
        const float* norm1_beta  = all_biases + s_offsets[0];
        const float* conv1_w     = all_weights + w_offsets[1];
        const float* conv1_s     = all_scales + s_offsets[1];  // ones (no BN, just bias)
        const float* conv1_b     = all_biases + s_offsets[1];
        const float* conv2_w     = all_weights + w_offsets[3];
        const float* conv2_s     = all_scales + s_offsets[3];
        const float* conv2_b     = all_biases + s_offsets[3];
        // Skip: RESIDUAL_ADD layer has cin, cout, nk (nk=0 means identity)
        const float* skip_w = (layers[4].cin != layers[4].cout) ? (all_weights + w_offsets[4]) : nullptr;
        const float* skip_b = (layers[4].cin != layers[4].cout) ? (all_biases + s_offsets[4]) : nullptr;

        // Output buffer (FP32) — size to actual N, not MAX_VOXELS.
        // GPU readback in gpu_fused_resblock writes exactly N*C_out bytes.
        outputs[0].set_shape(ov::Shape{(size_t)N, (size_t)C_out});
        float* out_ptr = outputs[0].data<float>();
        // No memset needed — gpu_fused_resblock writes all N*C_out values.

        gpu_fused_resblock(
            features_ptr, nmap_ptr, N, C_in, C_out,
            conv1_w, conv1_s, conv1_b, C_mid,
            conv2_w, conv2_s, conv2_b,
            norm1_gamma, norm1_beta,
            emb_scale_ptr, emb_shift_ptr,
            skip_w, skip_b,
            out_ptr,
            nmap_from_cache
        );

        return true;
    }

    float* cur_in  = buf_a_ptr;
    float* cur_out = buf_b_ptr;

    bool rebuild_nmap = false;

    // ── Process layers ──
    for (int li = 0; li < num_layers; li++) {
        auto& L = layers[li];
        const float* w = all_weights + w_offsets[li];
        const float* s = all_scales  + s_offsets[li];
        const float* b = all_biases  + s_offsets[li];

        if (rebuild_nmap) {
            // For strided/upsample layers, nmap changes — cannot use cache
            size_t nmap_size = (size_t)cur_N * 27;
            if (g_cached_nmap.size() < nmap_size)
                g_cached_nmap.resize(nmap_size);
            std::fill_n(g_cached_nmap.begin(), nmap_size, -1);
            nmap_ptr = g_cached_nmap.data();
            build_subm_neighbor_map(cur_coords.data(), cur_N, nmap_ptr);
            rebuild_nmap = false;
            // Invalidate cache since coords changed
            g_cached_nmap_N = 0;
        }

        switch (L.type) {
        case LAYER_SUBM_CONV:
        case LAYER_RES_CONV: {
            // GPU path: FP32→FP16 upload, SubMConv+BN on GPU, FP16→FP32 download
            gpu_subm_conv3d_full(cur_in, w, s, b, nmap_ptr,
                                 cur_N, L.cin, L.cout, cur_out);
            cpu_relu(cur_out, (size_t)cur_N * L.cout);
            std::swap(cur_in, cur_out);
            cur_C = L.cout;
            break;
        }
        case LAYER_STRIDED_DOWN: {
            // Save current state for later upsample
            SpatialCache cache;
            cache.coords.assign(cur_coords.data(),
                               cur_coords.data() + (size_t)cur_N * 4);
            cache.N = cur_N;

            // Compute new coordinates (stride=2 downsample)
            std::vector<int32_t> new_coords((size_t)m_max_voxels * 4, 0);
            std::vector<int32_t> in_to_out(cur_N, -1);
            int N_out = build_strided_conv_map(
                cur_coords.data(), cur_N, 2,
                new_coords.data(), in_to_out.data(), (int)m_max_voxels);

            // Save upsample reverse mapping
            cache.upsample_idx.resize(cur_N);
            std::copy(in_to_out.begin(), in_to_out.begin() + cur_N,
                     cache.upsample_idx.begin());
            ds_cache.push_back(std::move(cache));

            // Average pool features to output coords
            std::memset(cur_out, 0, (size_t)N_out * L.cout * sizeof(float));
            std::vector<int> counts(N_out, 0);
            for (int i = 0; i < cur_N; i++) {
                int j = in_to_out[i];
                if (j < 0) continue;
                // Apply conv weights first (strided conv = conv + stride)
                for (int co = 0; co < L.cout; co++) {
                    float acc = 0.0f;
                    for (int ci = 0; ci < L.cin; ci++) {
                        // Use center kernel element (13 = 1*9+1*3+1) for strided
                        acc += cur_in[(size_t)i * L.cin + ci] *
                               w[(size_t)13 * L.cin * L.cout + ci * L.cout + co];
                    }
                    cur_out[(size_t)j * L.cout + co] += acc;
                }
                counts[j]++;
            }
            // Average + BN
            for (int j = 0; j < N_out; j++) {
                if (counts[j] > 1) {
                    float inv = 1.0f / (float)counts[j];
                    for (int c = 0; c < L.cout; c++)
                        cur_out[(size_t)j * L.cout + c] *= inv;
                }
                for (int c = 0; c < L.cout; c++) {
                    cur_out[(size_t)j * L.cout + c] =
                        cur_out[(size_t)j * L.cout + c] * s[c] + b[c];
                }
            }
            cpu_relu(cur_out, (size_t)N_out * L.cout);

            // Update state
            std::memcpy(cur_coords.data(), new_coords.data(),
                       (size_t)N_out * 4 * sizeof(int32_t));
            cur_N = N_out;
            cur_C = L.cout;
            std::swap(cur_in, cur_out);
            rebuild_nmap = true;
            break;
        }
        case LAYER_INV_CONV_UP: {
            // Retrieve cached fine-resolution coords
            if (ds_cache.empty())
                throw std::runtime_error("SparseConv3dOp: upsample without prior downsample");
            auto& cache = ds_cache.back();

            // Nearest-neighbor upsample using cached mapping
            int N_fine = cache.N;
            std::memset(cur_out, 0, (size_t)N_fine * L.cout * sizeof(float));

            for (int i = 0; i < N_fine; i++) {
                int src = cache.upsample_idx[i];
                if (src < 0) continue;
                // Apply linear transform (inverse conv as linear + upsample)
                for (int co = 0; co < L.cout; co++) {
                    float acc = 0.0f;
                    for (int ci = 0; ci < L.cin; ci++) {
                        acc += cur_in[(size_t)src * L.cin + ci] *
                               w[ci * L.cout + co];
                    }
                    cur_out[(size_t)i * L.cout + co] = acc * s[co] + b[co];
                }
            }
            cpu_relu(cur_out, (size_t)N_fine * L.cout);

            // Restore fine-resolution coords
            std::memcpy(cur_coords.data(), cache.coords.data(),
                       (size_t)N_fine * 4 * sizeof(int32_t));
            cur_N = N_fine;
            cur_C = L.cout;
            ds_cache.pop_back();
            std::swap(cur_in, cur_out);
            rebuild_nmap = true;
            break;
        }
        case LAYER_LINEAR: {
            cpu_linear(cur_in, w, b, cur_N, L.cin, L.cout, cur_out);
            std::swap(cur_in, cur_out);
            cur_C = L.cout;
            break;
        }
        default:
            throw std::runtime_error("SparseConv3dOp: unknown layer type " +
                                     std::to_string(L.type));
        }
    }

    // Copy results to output tensor
    // Use cur_N (actual output voxels after strided convs) instead of MAX_VOXELS.
    outputs[0].set_shape(ov::Shape{(size_t)cur_N, (size_t)C_out});
    float* out_ptr = outputs[0].data<float>();
    std::memset(out_ptr, 0, (size_t)cur_N * C_out * sizeof(float));
    for (int i = 0; i < cur_N; i++) {
        std::memcpy(&out_ptr[(size_t)i * C_out],
                   &cur_in[(size_t)i * cur_C],
                   std::min(cur_C, C_out) * sizeof(float));
    }

    return true;
}

// ============================================================
// SparseMeshUpsampleOp — OV Operation Interface
// ============================================================

SparseMeshUpsampleOp::SparseMeshUpsampleOp(
    const ov::Output<ov::Node>& features,
    const ov::Output<ov::Node>& coords,
    const ov::Output<ov::Node>& num_voxels,
    const ov::Output<ov::Node>& params)
    : Op({features, coords, num_voxels, params})
{
    constructor_validate_and_infer_types();
}

SparseMeshUpsampleOp::SparseMeshUpsampleOp(const ov::OutputVector& args)
    : Op(args)
{
    constructor_validate_and_infer_types();
}

void SparseMeshUpsampleOp::validate_and_infer_types() {
    int64_t out_ch = m_dims.size() >= 8 ? m_dims[7] : 101;
    set_output_type(0, ov::element::f32, ov::PartialShape{ov::Dimension::dynamic(), out_ch});
    set_output_type(1, ov::element::i32, ov::PartialShape{ov::Dimension::dynamic(), 4});
    set_output_type(2, ov::element::i32, ov::PartialShape{1});
}

std::shared_ptr<ov::Node> SparseMeshUpsampleOp::clone_with_new_inputs(
    const ov::OutputVector& new_args) const
{
    auto op = std::make_shared<SparseMeshUpsampleOp>(new_args);
    op->m_max_out = m_max_out;
    op->m_dims = m_dims;
    return op;
}

bool SparseMeshUpsampleOp::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("max_out", m_max_out);
    visitor.on_attribute("dims", m_dims);
    return true;
}

bool SparseMeshUpsampleOp::has_evaluate() const { return true; }

bool SparseMeshUpsampleOp::evaluate(
    ov::TensorVector& outputs,
    const ov::TensorVector& inputs) const
{
    const float*   features_ptr = inputs[0].data<float>();
    const int32_t* coords_ptr   = inputs[1].data<int32_t>();
    const int32_t* num_ptr      = inputs[2].data<int32_t>();
    const float*   params_ptr   = inputs[3].data<float>();
    int N = num_ptr[0];

    int64_t out_ch = m_dims[7];
    // Nf = N * 8 * 8 (two subdivide blocks)
    int Nf_max = N * 64;

    // Allocate temp output, then copy into set_shape'd tensors
    std::vector<float>   out_feats((size_t)Nf_max * out_ch);
    std::vector<int32_t> out_coords((size_t)Nf_max * 4);

    int Nf = gpu_mesh_upsample(features_ptr, coords_ptr, N,
                               params_ptr, m_dims.data(),
                               out_feats.data(), out_coords.data());

    outputs[0].set_shape(ov::Shape{(size_t)Nf, (size_t)out_ch});
    outputs[1].set_shape(ov::Shape{(size_t)Nf, 4});
    outputs[2].set_shape(ov::Shape{1});
    std::memcpy(outputs[0].data<float>(),   out_feats.data(),  (size_t)Nf * out_ch * sizeof(float));
    std::memcpy(outputs[1].data<int32_t>(), out_coords.data(), (size_t)Nf * 4 * sizeof(int32_t));
    outputs[2].data<int32_t>()[0] = Nf;

    return true;
}

}  // namespace SAM3DExtension
