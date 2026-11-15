// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "details/cuda_type_traits.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

/**
 * @brief CUDA kernels for PagedAttentionExtension (M1 correctness kernels).
 *
 * KV cache layout (genai/OpenVINO default): both key_cache and value_cache are
 *   [num_blocks, num_kv_heads, block_size, head_size]   (row-major)
 *
 * Two kernels:
 *  - reshape_and_cache: scatter the new per-token K/V (ports 1/2) into the paged
 *    cache at positions given by `slot_mapping` (built from the block table).
 *  - paged_attention: one warp per (query token, query head); online-softmax over
 *    the causal KV horizon read from the paged cache. Handles prefill and decode
 *    in one launch (per-row causal length), GQA, distinct k/v head sizes, ALiBi,
 *    sliding window.
 *
 * See docs/design-rationales/PagedAttentionExtension.md §3. head_size and
 * v_head_size must be <= 256.
 */
struct PagedAttentionParams {
    int num_heads;       // query heads H
    int num_kv_heads;    // Hk
    int head_size;       // S  (key head size)
    int v_head_size;     // SV (value head size)
    int block_size;      // 32
    float scale;         // QK scale (defaults to 1/sqrt(S) at the op level)
    int sliding_window;  // 0 == disabled
    bool has_alibi;
};

/// Scatter new K/V into the paged cache. key/value/caches are `elem_type`.
void reshape_and_cache(cudaStream_t stream,
                       Type_t elem_type,
                       const void* key,
                       const void* value,
                       void* key_cache,
                       void* value_cache,
                       const std::int32_t* slot_mapping,
                       int num_tokens,
                       int num_kv_heads,
                       int head_size,
                       int v_head_size,
                       int block_size);

/// Paged attention. query/caches/out are `elem_type`; metadata is i32; alibi is f32.
void paged_attention(cudaStream_t stream,
                     Type_t elem_type,
                     const void* query,
                     const void* key_cache,
                     const void* value_cache,
                     const std::int32_t* block_indices,
                     const std::int32_t* block_indices_begins,
                     const std::int32_t* past_lens,
                     const std::int32_t* subsequence_begins,
                     const std::int32_t* row_to_subseq,
                     const float* alibi_slopes,
                     void* out,
                     const PagedAttentionParams& params,
                     int num_tokens);

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
