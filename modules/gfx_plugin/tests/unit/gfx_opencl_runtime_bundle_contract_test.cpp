// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <vector>

#include "backends/opencl/runtime/opencl_runtime_bundle.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

TEST(GfxOpenClRuntimeBundleContractTest,
     CandidateOrderPrefersPluginAdjacentOpenclBundle) {
    const auto candidates = opencl_runtime_library_candidates("/opt/openvino/lib");
    ASSERT_GE(candidates.size(), 9);
    EXPECT_EQ(candidates[0], "/opt/openvino/lib/opencl/libOpenCL.so");
    EXPECT_EQ(candidates[1], "/opt/openvino/lib/opencl/libOpenCL.so.1");
    EXPECT_EQ(candidates[2], "/opt/openvino/lib/opencl/libOpenCL.so.0.1");
    EXPECT_EQ(candidates[3], "/opt/openvino/lib/clvk/libOpenCL.so");
    EXPECT_EQ(candidates[6], "/opt/openvino/lib/libOpenCL.so");
}

TEST(GfxOpenClRuntimeBundleContractTest,
     BundleDescriptionMarksPluginAdjacentOpenclBundle) {
    const auto bundle =
        describe_opencl_runtime_bundle("/opt/openvino/lib/opencl/libOpenCL.so.0.1", "/opt/openvino/lib");
    EXPECT_TRUE(bundle.plugin_adjacent);
    EXPECT_EQ(bundle.bundle_dir, "/opt/openvino/lib/opencl");
    EXPECT_EQ(bundle.clspv_path, "/opt/openvino/lib/opencl/clspv");
    EXPECT_EQ(bundle.llvm_spirv_path, "/opt/openvino/lib/opencl/llvm-spirv");
}

TEST(GfxOpenClRuntimeBundleContractTest,
     SystemLibraryDoesNotAdvertiseBundledToolPaths) {
    const auto bundle =
        describe_opencl_runtime_bundle("/vendor/lib64/libOpenCL.so", "/opt/openvino/lib");
    EXPECT_FALSE(bundle.plugin_adjacent);
    EXPECT_TRUE(bundle.clspv_path.empty());
    EXPECT_TRUE(bundle.llvm_spirv_path.empty());
}

}  // namespace
}  // namespace gfx_plugin
}  // namespace ov
