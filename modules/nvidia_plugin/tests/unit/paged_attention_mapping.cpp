// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "ops/paged_attention_mapping.hpp"

using ov::nvidia_gpu::build_paged_attention_mapping;

namespace {
constexpr int kBlock = 4;
using V = std::vector<int32_t>;
}  // namespace

TEST(PagedAttentionMapping, SinglePrefillFromEmptyCache) {
    // 6 tokens, blocks [7, 2]: tokens 0-3 -> block 7, tokens 4-5 -> block 2.
    const auto m = build_paged_attention_mapping(6, kBlock, V{0}, V{0, 6}, V{7, 2}, V{0, 2});
    EXPECT_EQ(m.slot_mapping, (V{28, 29, 30, 31, 8, 9}));
    EXPECT_EQ(m.row_to_subseq, (V{0, 0, 0, 0, 0, 0}));
    EXPECT_EQ(m.write_blocks, (V{2, 7}));
    EXPECT_TRUE(m.refresh_blocks.empty());  // writes start at offset 0
}

TEST(PagedAttentionMapping, DecodeIntoPartiallyFilledBlockNeedsRefresh) {
    // past 5 -> position 5 is offset 1 of the 2nd block (id 3), which already holds position 4.
    const auto m = build_paged_attention_mapping(1, kBlock, V{5}, V{0, 1}, V{9, 3}, V{0, 2});
    EXPECT_EQ(m.slot_mapping, (V{13}));
    EXPECT_EQ(m.write_blocks, (V{3}));
    EXPECT_EQ(m.refresh_blocks, (V{3}));
}

TEST(PagedAttentionMapping, DecodeAtBlockBoundaryNeedsNoRefresh) {
    const auto m = build_paged_attention_mapping(1, kBlock, V{4}, V{0, 1}, V{9, 3}, V{0, 2});
    EXPECT_EQ(m.slot_mapping, (V{12}));
    EXPECT_TRUE(m.refresh_blocks.empty());
}

TEST(PagedAttentionMapping, MixedBatchWithSharedAndPermutedBlocks) {
    // seq 0: decode, past 6, blocks [5, 1] (block 5 is a full prefix block shared with seq 2);
    // seq 1: chunked prefill, past 2, 3 tokens, blocks [3, 8];
    // seq 2: decode crossing into a new block, past 8, blocks [5, 4, 6].
    const auto m = build_paged_attention_mapping(5,
                                                 kBlock,
                                                 V{6, 2, 8},
                                                 V{0, 1, 4, 5},
                                                 V{5, 1, 3, 8, 5, 4, 6},
                                                 V{0, 2, 4, 7});
    EXPECT_EQ(m.slot_mapping, (V{1 * 4 + 2, 3 * 4 + 2, 3 * 4 + 3, 8 * 4 + 0, 6 * 4 + 0}));
    EXPECT_EQ(m.row_to_subseq, (V{0, 1, 1, 1, 2}));
    EXPECT_EQ(m.write_blocks, (V{1, 3, 6, 8}));
    EXPECT_EQ(m.refresh_blocks, (V{1, 3}));
}

TEST(PagedAttentionMapping, SequenceWithoutNewTokensIsIgnored) {
    const auto m = build_paged_attention_mapping(1, kBlock, V{3, 5}, V{0, 0, 1}, V{2, 3, 4}, V{0, 1, 3});
    EXPECT_EQ(m.slot_mapping, (V{4 * 4 + 1}));
    EXPECT_EQ(m.refresh_blocks, (V{4}));
}

TEST(PagedAttentionMapping, NegativeBlockIdsAreNotWritten) {
    const auto m = build_paged_attention_mapping(2, kBlock, V{0}, V{0, 2}, V{-1}, V{0, 1});
    EXPECT_EQ(m.slot_mapping, (V{-1, -1}));
    EXPECT_TRUE(m.write_blocks.empty());
}

TEST(PagedAttentionMapping, InvalidMetadataThrows) {
    // Wrong subsequence_begins size.
    EXPECT_ANY_THROW(build_paged_attention_mapping(1, kBlock, V{0}, V{0}, V{0}, V{0, 1}));
    // subsequence_begins does not cover all tokens.
    EXPECT_ANY_THROW(build_paged_attention_mapping(2, kBlock, V{0}, V{0, 1}, V{0}, V{0, 1}));
    // Not enough blocks for the positions.
    EXPECT_ANY_THROW(build_paged_attention_mapping(5, kBlock, V{0}, V{0, 5}, V{0}, V{0, 1}));
    // Block id out of the cache (written block, and a block only read as context).
    EXPECT_ANY_THROW(build_paged_attention_mapping(1, kBlock, V{0}, V{0, 1}, V{10}, V{0, 1}, 10));
    EXPECT_ANY_THROW(build_paged_attention_mapping(1, kBlock, V{4}, V{0, 1}, V{10, 2}, V{0, 2}, 10));
    // block_indices_begins out of range.
    EXPECT_ANY_THROW(build_paged_attention_mapping(1, kBlock, V{0}, V{0, 1}, V{0}, V{0, 2}));
}
