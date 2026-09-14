// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "runtime/immutable_gpu_buffer_cache.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

GpuBuffer make_test_buffer(uint64_t uid, size_t bytes = 16) {
    GpuBuffer buffer{};
    buffer.buffer = reinterpret_cast<void*>(static_cast<uintptr_t>(uid));
    buffer.size = bytes;
    buffer.type = ov::element::u8;
    buffer.allocation_uid = uid;
    return buffer;
}

TEST(GpuConstCacheTest, ReusesIdenticalPayloadsForSameLogicalKey) {
    int create_calls = 0;
    ImmutableGpuBufferCache cache;
    const uint8_t payload[] = {1, 2, 3, 4};

    const auto first = cache.get_or_create("weights", payload, sizeof(payload), ov::element::u8, [&]() {
        ++create_calls;
        return make_test_buffer(11);
    });
    const auto second = cache.get_or_create("weights", payload, sizeof(payload), ov::element::u8, [&]() {
        ++create_calls;
        return make_test_buffer(22);
    });

    EXPECT_EQ(create_calls, 1);
    EXPECT_EQ(first.allocation_uid, second.allocation_uid);
    EXPECT_EQ(cache.entry_count(), 1u);
    EXPECT_EQ(cache.total_bytes(), sizeof(payload));
}

TEST(GpuConstCacheTest, SeparatesDifferentPayloadsUnderSameLogicalKey) {
    int create_calls = 0;
    ImmutableGpuBufferCache cache;
    const uint8_t lhs[] = {1, 2, 3, 4};
    const uint8_t rhs[] = {1, 2, 3, 5};

    const auto first = cache.get_or_create("weights", lhs, sizeof(lhs), ov::element::u8, [&]() {
        ++create_calls;
        return make_test_buffer(101);
    });
    const auto second = cache.get_or_create("weights", rhs, sizeof(rhs), ov::element::u8, [&]() {
        ++create_calls;
        return make_test_buffer(202);
    });

    EXPECT_EQ(create_calls, 2);
    EXPECT_NE(first.allocation_uid, second.allocation_uid);
    EXPECT_EQ(cache.entry_count(), 2u);
    EXPECT_EQ(cache.total_bytes(), sizeof(lhs) + sizeof(rhs));
}

}  // namespace
}  // namespace gfx_plugin
}  // namespace ov
