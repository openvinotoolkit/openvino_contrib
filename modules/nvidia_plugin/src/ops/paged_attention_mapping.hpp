// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "openvino/core/except.hpp"

namespace ov {
namespace nvidia_gpu {

/**
 * @brief Host-side view of one PagedAttentionExtension step, derived from the
 *        scheduling metadata (past_lens, subsequence_begins, block_indices,
 *        block_indices_begins).
 */
struct PagedAttentionMapping {
    /// Cache slot (block * block_size + offset) each query row writes its K/V to; -1 = no write.
    std::vector<int32_t> slot_mapping;
    /// Subsequence index of each query row.
    std::vector<int32_t> row_to_subseq;
    /// Distinct physical blocks written this step (sorted). They must be written back to the
    /// host cache tensor so the host stays the source of truth.
    std::vector<int32_t> write_blocks;
    /// Distinct physical blocks that are appended to at a non-zero in-block offset (sorted).
    /// Their already-filled prefix may have been produced outside the model (e.g. GenAI's
    /// copy-on-write `copy_blocks` on the host tensor), so they must be refreshed from the
    /// host before this step writes into them.
    std::vector<int32_t> refresh_blocks;
};

namespace paged_attention_detail {
inline void sort_unique(std::vector<int32_t>& v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}
}  // namespace paged_attention_detail

/**
 * @brief Builds and validates the per-step mapping.
 * @param num_cache_blocks Number of blocks in the bound cache tensor (dim 0), or -1 if unknown.
 * Throws on inconsistent metadata instead of reading or writing out of bounds.
 */
inline PagedAttentionMapping build_paged_attention_mapping(int num_tokens,
                                                           int block_size,
                                                           const std::vector<int32_t>& past_lens,
                                                           const std::vector<int32_t>& subsequence_begins,
                                                           const std::vector<int32_t>& block_indices,
                                                           const std::vector<int32_t>& block_indices_begins,
                                                           int64_t num_cache_blocks = -1) {
    OPENVINO_ASSERT(block_size > 0, "PagedAttention: block_size must be positive");
    OPENVINO_ASSERT(num_tokens >= 0, "PagedAttention: negative number of tokens");
    const int nseq = static_cast<int>(past_lens.size());
    OPENVINO_ASSERT(subsequence_begins.size() == static_cast<size_t>(nseq) + 1,
                    "PagedAttention: subsequence_begins must have ",
                    nseq + 1,
                    " elements, got ",
                    subsequence_begins.size());
    OPENVINO_ASSERT(block_indices_begins.size() == static_cast<size_t>(nseq) + 1,
                    "PagedAttention: block_indices_begins must have ",
                    nseq + 1,
                    " elements, got ",
                    block_indices_begins.size());
    OPENVINO_ASSERT(subsequence_begins.front() == 0 && subsequence_begins.back() == num_tokens,
                    "PagedAttention: subsequence_begins must span [0, ",
                    num_tokens,
                    "]");
    OPENVINO_ASSERT(block_indices_begins.front() == 0 &&
                        block_indices_begins.back() <= static_cast<int32_t>(block_indices.size()),
                    "PagedAttention: block_indices_begins is out of range of block_indices");

    PagedAttentionMapping m;
    m.slot_mapping.assign(num_tokens, -1);
    m.row_to_subseq.assign(num_tokens, -1);

    const auto check_block = [&](int32_t bn) {
        OPENVINO_ASSERT(num_cache_blocks < 0 || bn < num_cache_blocks,
                        "PagedAttention: block index ",
                        bn,
                        " is out of range of the KV cache with ",
                        num_cache_blocks,
                        " blocks");
    };

    for (int i = 0; i < nseq; ++i) {
        const int row_begin = subsequence_begins[i];
        const int row_end = subsequence_begins[i + 1];
        const int blk_begin = block_indices_begins[i];
        const int blk_end = block_indices_begins[i + 1];
        const int past = past_lens[i];
        OPENVINO_ASSERT(row_begin <= row_end, "PagedAttention: subsequence_begins must be non-decreasing");
        OPENVINO_ASSERT(blk_begin <= blk_end, "PagedAttention: block_indices_begins must be non-decreasing");
        OPENVINO_ASSERT(past >= 0, "PagedAttention: negative past_lens[", i, "]");

        if (row_end > row_begin) {
            // The attention kernel reads every block up to the last new token.
            const int num_blocks = (past + (row_end - row_begin) + block_size - 1) / block_size;
            OPENVINO_ASSERT(blk_begin + num_blocks <= blk_end,
                            "PagedAttention: sequence ", i, " has ", blk_end - blk_begin,
                            " blocks, but its ", past + (row_end - row_begin), " tokens need ", num_blocks);
            for (int b = blk_begin; b < blk_begin + num_blocks; ++b) {
                if (block_indices[b] >= 0) {
                    check_block(block_indices[b]);
                }
            }
        }
        for (int row = row_begin; row < row_end; ++row) {
            m.row_to_subseq[row] = i;
            const int pos = past + (row - row_begin);
            const int blk = blk_begin + pos / block_size;
            OPENVINO_ASSERT(blk < blk_end,
                            "PagedAttention: sequence ",
                            i,
                            " has no block for token position ",
                            pos);
            const int32_t bn = block_indices[blk];
            if (bn < 0) {
                continue;  // padding: nothing to write
            }
            check_block(bn);
            m.slot_mapping[row] = bn * block_size + (pos % block_size);
            m.write_blocks.push_back(bn);
        }

        // The block this subsequence appends into already holds `past % block_size` tokens.
        if (row_end > row_begin && past % block_size != 0) {
            const int blk = blk_begin + past / block_size;
            OPENVINO_ASSERT(blk < blk_end, "PagedAttention: sequence ", i, " has no block for position ", past);
            const int32_t bn = block_indices[blk];
            if (bn >= 0) {
                check_block(bn);
                m.refresh_blocks.push_back(bn);
            }
        }
    }
    paged_attention_detail::sort_unique(m.write_blocks);
    paged_attention_detail::sort_unique(m.refresh_blocks);
    return m;
}

}  // namespace nvidia_gpu
}  // namespace ov
