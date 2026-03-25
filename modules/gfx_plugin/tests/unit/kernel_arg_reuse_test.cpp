// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "kernel_ir/gfx_kernel_args.hpp"
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

class TestConstBufferManager final : public GpuBufferManager {
public:
    TestConstBufferManager() = default;

    bool supports_const_cache() const override {
        return true;
    }

    GpuBuffer wrap_const(const std::string& key,
                         const void* data,
                         size_t bytes,
                         ov::element::Type type) override {
        return m_cache.get_or_create(key, data, bytes, type, [&]() {
            ++m_create_calls;
            return make_test_buffer(static_cast<uint64_t>(1000 + m_create_calls), bytes);
        });
    }

    int create_calls() const {
        return m_create_calls;
    }

private:
    ImmutableGpuBufferCache m_cache;
    int m_create_calls = 0;
};

TEST(KernelArgReuseTest, ReusesBytesArgsAcrossDifferentStageNames) {
    TestConstBufferManager buffer_manager;
    const uint32_t payload = 0x01020304u;
    const std::vector<KernelArg> args = {make_bytes_arg(0, &payload, sizeof(payload))};

    const auto first = materialize_kernel_bytes_args(args, buffer_manager, "stage_a");
    const auto second = materialize_kernel_bytes_args(args, buffer_manager, "stage_b");

    ASSERT_EQ(first.size(), 1u);
    ASSERT_EQ(second.size(), 1u);
    EXPECT_EQ(buffer_manager.create_calls(), 1);
    EXPECT_EQ(first[0].kind, KernelArg::Kind::Buffer);
    EXPECT_EQ(second[0].kind, KernelArg::Kind::Buffer);
    EXPECT_EQ(first[0].buffer.allocation_uid, second[0].buffer.allocation_uid);
}

TEST(KernelArgReuseTest, SeparatesDistinctBytesPayloadsAcrossStages) {
    TestConstBufferManager buffer_manager;
    const uint32_t lhs = 0x01020304u;
    const uint32_t rhs = 0x01020305u;

    const auto first =
        materialize_kernel_bytes_args({make_bytes_arg(0, &lhs, sizeof(lhs))}, buffer_manager, "stage_a");
    const auto second =
        materialize_kernel_bytes_args({make_bytes_arg(0, &rhs, sizeof(rhs))}, buffer_manager, "stage_b");

    ASSERT_EQ(first.size(), 1u);
    ASSERT_EQ(second.size(), 1u);
    EXPECT_EQ(buffer_manager.create_calls(), 2);
    EXPECT_NE(first[0].buffer.allocation_uid, second[0].buffer.allocation_uid);
}

}  // namespace
}  // namespace gfx_plugin
}  // namespace ov
