// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/core/except.hpp"

#include "backends/vulkan/codegen/vulkan_codegen_backend.hpp"
#include "backends/vulkan/runtime/vulkan_buffer_manager.hpp"
#include "mlir/spirv_codegen.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

TEST(VulkanBackendTest, CompileToSpirv) {
    VulkanCodegenBackend backend;
    KernelSource src;
    src.entry_point = "gfx_stub";

    std::string log;
    EXPECT_THROW(backend.compile(src, &log), ov::Exception);
}

TEST(VulkanBackendTest, CompileDoesNotReuseMutableKernelInstance) {
    VulkanCodegenBackend backend;
    std::string log;

    KernelSource src;
    src.entry_point = "gfx_stub";
    src.signature.arg_count = 0;
    src.spirv_binary = build_stub_spirv("gfx_stub", &log);
    ASSERT_FALSE(src.spirv_binary.empty()) << log;

    auto first = backend.compile(src);
    auto second = backend.compile(src);
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    EXPECT_NE(first.get(), second.get());
}

TEST(VulkanBackendTest, ConstCacheSeparatesDifferentPayloads) {
    VulkanBufferManager buffer_manager;

    const uint32_t a[4] = {1u, 2u, 3u, 4u};
    const uint32_t b[4] = {4u, 3u, 2u, 1u};

    GpuBuffer first = buffer_manager.wrap_const("shape_params", a, sizeof(a), ov::element::u32);
    GpuBuffer first_again = buffer_manager.wrap_const("shape_params", a, sizeof(a), ov::element::u32);
    GpuBuffer second = buffer_manager.wrap_const("shape_params", b, sizeof(b), ov::element::u32);

    ASSERT_TRUE(first.valid());
    ASSERT_TRUE(first_again.valid());
    ASSERT_TRUE(second.valid());
    EXPECT_EQ(first.buffer, first_again.buffer);
    EXPECT_NE(first.buffer, second.buffer);
}

TEST(VulkanBackendTest, SmallConstUsesDirectMappedStorage) {
    VulkanBufferManager buffer_manager;

    const uint32_t values[4] = {1u, 2u, 3u, 4u};
    GpuBuffer buf = buffer_manager.wrap_const("small_params", values, sizeof(values), ov::element::u32);

    ASSERT_TRUE(buf.valid());
    EXPECT_TRUE(buf.host_visible);
}

}  // namespace
}  // namespace gfx_plugin
}  // namespace ov
