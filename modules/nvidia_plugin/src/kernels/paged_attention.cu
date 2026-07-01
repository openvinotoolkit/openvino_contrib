// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_fp16.h>
#include <fmt/format.h>

#include <algorithm>
#include <cmath>

#include "details/error.hpp"
#include "paged_attention.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

namespace {

// Element access helpers (cache/query stored as T, math done in float).
__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ void from_float(float& dst, float v) { dst = v; }
__device__ __forceinline__ void from_float(__half& dst, float v) { dst = __float2half(v); }

constexpr int kMaxPerLane = 8;  // supports head_size <= 256 (8 * 32 lanes)

template <typename T>
__global__ void reshape_and_cache_kernel(const T* __restrict__ key,
                                         const T* __restrict__ value,
                                         T* __restrict__ key_cache,
                                         T* __restrict__ value_cache,
                                         const int* __restrict__ slot_mapping,
                                         int num_tokens,
                                         int num_kv_heads,
                                         int head_size,
                                         int v_head_size,
                                         int block_size) {
    const int token = blockIdx.x;
    const int hk = blockIdx.y;
    const int lane = threadIdx.x;
    if (token >= num_tokens || hk >= num_kv_heads) return;
    const int slot = slot_mapping[token];
    if (slot < 0) return;
    const int block = slot / block_size;
    const int offset = slot % block_size;
    if (lane < head_size) {
        const T v = key[token * (num_kv_heads * head_size) + hk * head_size + lane];
        const size_t idx = ((size_t)block * num_kv_heads + hk) * block_size * head_size +
                           (size_t)offset * head_size + lane;
        key_cache[idx] = v;
    }
    if (lane < v_head_size) {
        const T v = value[token * (num_kv_heads * v_head_size) + hk * v_head_size + lane];
        const size_t idx = ((size_t)block * num_kv_heads + hk) * block_size * v_head_size +
                           (size_t)offset * v_head_size + lane;
        value_cache[idx] = v;
    }
}

// One warp (32 lanes) per (query token, query head). Online softmax over the causal
// KV horizon read from the paged cache.
template <typename T>
__global__ void paged_attention_kernel(const T* __restrict__ query,
                                       const T* __restrict__ key_cache,
                                       const T* __restrict__ value_cache,
                                       const int* __restrict__ block_indices,
                                       const int* __restrict__ block_indices_begins,
                                       const int* __restrict__ past_lens,
                                       const int* __restrict__ subsequence_begins,
                                       const int* __restrict__ row_to_subseq,
                                       const float* __restrict__ alibi_slopes,
                                       T* __restrict__ out,
                                       PagedAttentionParams p,
                                       int num_tokens) {
    const int row = blockIdx.x;
    const int h = blockIdx.y;
    const int lane = threadIdx.x;
    if (row >= num_tokens || h >= p.num_heads) return;

    const int i = row_to_subseq[row];
    const int hk = h / (p.num_heads / p.num_kv_heads);
    const int row_in_subseq = row - subsequence_begins[i];
    const int cur_kv_len = past_lens[i] + row_in_subseq + 1;  // causal horizon
    const int S = p.head_size, SV = p.v_head_size, BS = p.block_size;
    const float alibi = p.has_alibi ? alibi_slopes[h] : 0.0f;

    float qreg[kMaxPerLane];
    for (int t = 0; t < kMaxPerLane; ++t) {
        const int d = lane + t * 32;
        qreg[t] = (d < S) ? to_float(query[row * (p.num_heads * S) + h * S + d]) * p.scale : 0.0f;
    }

    float m = -1e30f, l = 0.0f;
    float acc[kMaxPerLane];
    for (int t = 0; t < kMaxPerLane; ++t) acc[t] = 0.0f;

    const int blk_begin = block_indices_begins[i];
    for (int pcur = 0; pcur < cur_kv_len; ++pcur) {
        if (p.sliding_window > 0 && pcur < cur_kv_len - p.sliding_window) continue;
        const int bn = block_indices[blk_begin + pcur / BS];
        if (bn < 0) continue;
        const int off = pcur % BS;
        const size_t kbase = ((size_t)bn * p.num_kv_heads + hk) * BS * S + (size_t)off * S;
        float partial = 0.0f;
        for (int t = 0; t < kMaxPerLane; ++t) {
            const int d = lane + t * 32;
            if (d < S) partial += qreg[t] * to_float(key_cache[kbase + d]);
        }
        for (int o = 16; o > 0; o >>= 1) partial += __shfl_xor_sync(0xffffffff, partial, o);
        float s = partial;  // query already scaled
        if (p.has_alibi) s += alibi * (float)(-(cur_kv_len - 1 - pcur));
        const float m_new = fmaxf(m, s);
        const float corr = __expf(m - m_new);
        const float e = __expf(s - m_new);
        l = l * corr + e;
        const size_t vbase = ((size_t)bn * p.num_kv_heads + hk) * BS * SV + (size_t)off * SV;
        for (int t = 0; t < kMaxPerLane; ++t) {
            const int d = lane + t * 32;
            if (d < SV) acc[t] = acc[t] * corr + e * to_float(value_cache[vbase + d]);
        }
        m = m_new;
    }
    const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
    for (int t = 0; t < kMaxPerLane; ++t) {
        const int d = lane + t * 32;
        if (d < SV) from_float(out[row * (p.num_heads * SV) + h * SV + d], acc[t] * inv);
    }
}

template <typename T>
void launch_reshape_and_cache(cudaStream_t stream, const void* key, const void* value, void* key_cache,
                              void* value_cache, const int* slot_mapping, int num_tokens, int num_kv_heads,
                              int head_size, int v_head_size, int block_size) {
    if (num_tokens == 0) return;
    dim3 grid(num_tokens, num_kv_heads);
    const unsigned threads = static_cast<unsigned>(std::max(head_size, v_head_size));
    reshape_and_cache_kernel<T><<<grid, threads, 0, stream>>>(
        static_cast<const T*>(key), static_cast<const T*>(value), static_cast<T*>(key_cache),
        static_cast<T*>(value_cache), slot_mapping, num_tokens, num_kv_heads, head_size, v_head_size, block_size);
}

template <typename T>
void launch_paged_attention(cudaStream_t stream, const void* query, const void* key_cache, const void* value_cache,
                            const int* block_indices, const int* block_indices_begins, const int* past_lens,
                            const int* subsequence_begins, const int* row_to_subseq, const float* alibi_slopes,
                            void* out, const PagedAttentionParams& p, int num_tokens) {
    if (num_tokens == 0) return;
    dim3 grid(num_tokens, p.num_heads);
    paged_attention_kernel<T><<<grid, 32, 0, stream>>>(
        static_cast<const T*>(query), static_cast<const T*>(key_cache), static_cast<const T*>(value_cache),
        block_indices, block_indices_begins, past_lens, subsequence_begins, row_to_subseq, alibi_slopes,
        static_cast<T*>(out), p, num_tokens);
}

}  // namespace

void reshape_and_cache(cudaStream_t stream, Type_t elem_type, const void* key, const void* value, void* key_cache,
                       void* value_cache, const std::int32_t* slot_mapping, int num_tokens, int num_kv_heads,
                       int head_size, int v_head_size, int block_size) {
    if (head_size > kMaxPerLane * 32 || v_head_size > kMaxPerLane * 32) {
        throw_ov_exception("PagedAttention: head_size/v_head_size must be <= 256");
    }
    switch (elem_type) {
        case Type_t::f32:
            launch_reshape_and_cache<float>(stream, key, value, key_cache, value_cache, slot_mapping, num_tokens,
                                            num_kv_heads, head_size, v_head_size, block_size);
            break;
        case Type_t::f16:
            launch_reshape_and_cache<__half>(stream, key, value, key_cache, value_cache, slot_mapping, num_tokens,
                                             num_kv_heads, head_size, v_head_size, block_size);
            break;
        default:
            throw_ov_exception(fmt::format("PagedAttention reshape_and_cache: unsupported element type {}",
                                           static_cast<int>(elem_type)));
    }
    throwIfError(cudaPeekAtLastError());
}

void paged_attention(cudaStream_t stream, Type_t elem_type, const void* query, const void* key_cache,
                     const void* value_cache, const std::int32_t* block_indices,
                     const std::int32_t* block_indices_begins, const std::int32_t* past_lens,
                     const std::int32_t* subsequence_begins, const std::int32_t* row_to_subseq,
                     const float* alibi_slopes, void* out, const PagedAttentionParams& params, int num_tokens) {
    if (params.head_size > kMaxPerLane * 32 || params.v_head_size > kMaxPerLane * 32) {
        throw_ov_exception("PagedAttention: head_size/v_head_size must be <= 256");
    }
    if (params.num_kv_heads <= 0 || params.num_heads % params.num_kv_heads != 0) {
        throw_ov_exception("PagedAttention: num_heads must be a multiple of num_kv_heads");
    }
    switch (elem_type) {
        case Type_t::f32:
            launch_paged_attention<float>(stream, query, key_cache, value_cache, block_indices, block_indices_begins,
                                          past_lens, subsequence_begins, row_to_subseq, alibi_slopes, out, params,
                                          num_tokens);
            break;
        case Type_t::f16:
            launch_paged_attention<__half>(stream, query, key_cache, value_cache, block_indices, block_indices_begins,
                                           past_lens, subsequence_begins, row_to_subseq, alibi_slopes, out, params,
                                           num_tokens);
            break;
        default:
            throw_ov_exception(
                fmt::format("PagedAttention: unsupported element type {}", static_cast<int>(elem_type)));
    }
    throwIfError(cudaPeekAtLastError());
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
