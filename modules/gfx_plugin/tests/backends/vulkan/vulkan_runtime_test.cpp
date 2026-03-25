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

TEST(VulkanBackendTest, CompileForksKernelInstanceButReusesImmutableProgram) {
    VulkanCodegenBackend backend;
    std::string log;

    KernelSource src;
    src.entry_point = "gfx_stub";
    src.signature.arg_count = 1;
    src.spirv_binary = build_stub_spirv("gfx_stub", &log);
    ASSERT_FALSE(src.spirv_binary.empty()) << log;

    auto first = backend.compile(src);
    auto second = backend.compile(src);
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    EXPECT_NE(first.get(), second.get());

    auto* first_vk = dynamic_cast<VulkanCompiledKernel*>(first.get());
    auto* second_vk = dynamic_cast<VulkanCompiledKernel*>(second.get());
    ASSERT_NE(first_vk, nullptr);
    ASSERT_NE(second_vk, nullptr);
    EXPECT_EQ(first_vk->shared_program_identity(), second_vk->shared_program_identity());
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

TEST(VulkanBackendTest, ConstBufferReuseContextIsSharedAcrossManagers) {
    VulkanBufferManager first_manager;
    VulkanBufferManager second_manager;

    EXPECT_EQ(first_manager.shared_const_cache_identity(), second_manager.shared_const_cache_identity());

    const uint32_t values[4] = {1u, 2u, 3u, 4u};
    GpuBuffer first = first_manager.wrap_const("shared_weights", values, sizeof(values), ov::element::u32);
    GpuBuffer second = second_manager.wrap_const("shared_weights", values, sizeof(values), ov::element::u32);

    ASSERT_TRUE(first.valid());
    ASSERT_TRUE(second.valid());
    EXPECT_EQ(first.buffer, second.buffer);
}

TEST(VulkanBackendTest, DescriptorCacheReusesEquivalentBindings) {
    std::string log;
    VulkanCodegenBackend backend;

    KernelSource src;
    src.entry_point = "gfx_stub";
    src.signature.arg_count = 0;
    src.spirv_binary = build_stub_spirv("gfx_stub", &log);
    ASSERT_FALSE(src.spirv_binary.empty()) << log;

    auto kernel = backend.compile(src);
    ASSERT_TRUE(kernel);
    auto* vk_kernel = dynamic_cast<VulkanCompiledKernel*>(kernel.get());
    ASSERT_NE(vk_kernel, nullptr);

    KernelDispatch dispatch;
    dispatch.grid[0] = 1;
    dispatch.threads_per_group[0] = 1;

    std::vector<KernelArg> args;
    kernel->execute(nullptr, dispatch, args, nullptr);
    EXPECT_EQ(vk_kernel->cached_descriptor_set_count(), 1u);

    kernel->execute(nullptr, dispatch, args, nullptr);
    EXPECT_EQ(vk_kernel->cached_descriptor_set_count(), 1u);
}

TEST(VulkanBackendTest, DescriptorCacheIsSharedAcrossForkedKernelInstances) {
    std::string log;
    VulkanCodegenBackend backend;
    VulkanBufferManager buffer_manager;

    KernelSource src;
    src.entry_point = "gfx_stub";
    src.signature.arg_count = 1;
    src.spirv_binary = build_stub_spirv("gfx_stub", &log);
    ASSERT_FALSE(src.spirv_binary.empty()) << log;

    auto first = backend.compile(src);
    auto second = backend.compile(src);
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);

    auto* first_vk = dynamic_cast<VulkanCompiledKernel*>(first.get());
    auto* second_vk = dynamic_cast<VulkanCompiledKernel*>(second.get());
    ASSERT_NE(first_vk, nullptr);
    ASSERT_NE(second_vk, nullptr);

    KernelDispatch dispatch;
    dispatch.grid[0] = 1;
    dispatch.threads_per_group[0] = 1;

    const uint32_t values[4] = {1u, 2u, 3u, 4u};
    GpuBuffer buf = buffer_manager.wrap_const("descriptor_cache_shared", values, sizeof(values), ov::element::u32);
    ASSERT_TRUE(buf.valid());

    std::vector<KernelArg> args = {make_buffer_arg(0, buf)};
    first->execute(nullptr, dispatch, args, nullptr);
    EXPECT_EQ(first_vk->cached_descriptor_set_count(), 1u);
    EXPECT_EQ(second_vk->cached_descriptor_set_count(), 1u);

    second->execute(nullptr, dispatch, args, nullptr);
    EXPECT_EQ(first_vk->cached_descriptor_set_count(), 1u);
    EXPECT_EQ(second_vk->cached_descriptor_set_count(), 1u);
}

TEST(VulkanBackendTest, BindingSchemaAndDescriptorCacheAreSharedAcrossDistinctProgramsWithSameAbi) {
    std::string log;
    VulkanCodegenBackend backend;
    VulkanBufferManager buffer_manager;

    KernelSource first_src;
    first_src.entry_point = "gfx_stub_a";
    first_src.signature.arg_count = 1;
    first_src.spirv_binary = build_stub_spirv("gfx_stub_a", &log);
    ASSERT_FALSE(first_src.spirv_binary.empty()) << log;

    KernelSource second_src;
    second_src.entry_point = "gfx_stub_b";
    second_src.signature.arg_count = 1;
    second_src.spirv_binary = build_stub_spirv("gfx_stub_b", &log);
    ASSERT_FALSE(second_src.spirv_binary.empty()) << log;

    auto first = backend.compile(first_src);
    auto second = backend.compile(second_src);
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);

    auto* first_vk = dynamic_cast<VulkanCompiledKernel*>(first.get());
    auto* second_vk = dynamic_cast<VulkanCompiledKernel*>(second.get());
    ASSERT_NE(first_vk, nullptr);
    ASSERT_NE(second_vk, nullptr);
    EXPECT_NE(first_vk->shared_program_identity(), second_vk->shared_program_identity());
    EXPECT_EQ(first_vk->shared_binding_schema_identity(), second_vk->shared_binding_schema_identity());

    KernelDispatch dispatch;
    dispatch.grid[0] = 1;
    dispatch.threads_per_group[0] = 1;

    const uint32_t values[4] = {1u, 2u, 3u, 4u};
    GpuBuffer buf = buffer_manager.wrap_const("descriptor_cache_schema_shared", values, sizeof(values), ov::element::u32);
    ASSERT_TRUE(buf.valid());

    std::vector<KernelArg> args = {make_buffer_arg(0, buf)};
    first->execute(nullptr, dispatch, args, nullptr);
    EXPECT_EQ(first_vk->cached_descriptor_set_count(), 1u);
    EXPECT_EQ(second_vk->cached_descriptor_set_count(), 1u);

    second->execute(nullptr, dispatch, args, nullptr);
    EXPECT_EQ(first_vk->cached_descriptor_set_count(), 1u);
    EXPECT_EQ(second_vk->cached_descriptor_set_count(), 1u);
}

}  // namespace
}  // namespace gfx_plugin
}  // namespace ov
