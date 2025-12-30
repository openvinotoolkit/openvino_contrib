// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/core/except.hpp"

#include "backends/vulkan/codegen/vulkan_compiler.hpp"

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

}  // namespace
}  // namespace gfx_plugin
}  // namespace ov
