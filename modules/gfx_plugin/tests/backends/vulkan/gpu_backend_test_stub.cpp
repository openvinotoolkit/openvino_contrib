// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "mlir/spirv_codegen.hpp"
#include "backends/vulkan/codegen/vulkan_codegen_backend.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

TEST(GfxBackendTest, CompileAndExecuteKernel) {
    std::string log;
    auto spirv = build_stub_spirv("gfx_stub", &log);
    ASSERT_FALSE(spirv.empty()) << log;

    auto kernel = std::make_shared<VulkanCompiledKernel>(std::move(spirv), "gfx_stub", 0);
    ASSERT_TRUE(kernel);

    KernelDispatch dispatch;
    dispatch.grid[0] = 1;
    dispatch.threads_per_group[0] = 1;

    std::vector<KernelArg> args;
    kernel->execute(nullptr, dispatch, args, nullptr);
    EXPECT_EQ(kernel->args_count(), 0u);
}

}  // namespace
}  // namespace gfx_plugin
}  // namespace ov
