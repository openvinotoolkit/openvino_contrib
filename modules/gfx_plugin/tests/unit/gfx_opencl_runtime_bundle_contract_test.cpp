// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <vector>

#include "backends/opencl/runtime/opencl_program_cache.hpp"
#include "backends/opencl/runtime/opencl_runtime_bundle.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

OpenClProgramBuildRequest make_contract_program_request() {
    OpenClProgramBuildRequest request;
    request.manifest_ref = "manifest://unit/opencl/eltwise";
    request.abi_fingerprint = "abi://unit/opencl/eltwise";
    request.artifact_key = "artifact://unit/opencl/eltwise";
    request.backend_domain = "opencl";
    request.kernel_id = "opencl/generated/eltwise_f32";
    request.stage_record_key = 0x1234u;
    request.source_id = "opencl/generated/eltwise_f32";
    request.entry_point = "gfx_eltwise_f32";
    request.compile_options_key = "compile-options://unit/opencl/default";
    request.build_options = "-cl-std=CL1.2";
    request.source = "__kernel void gfx_eltwise_f32(__global float* out) { out[0] = 1.0f; }";
    return request;
}

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

TEST(GfxOpenClRuntimeBundleContractTest,
     ProgramCacheKeyUsesCompilerOwnedDescriptorIdentity) {
    const auto base = make_contract_program_request();
    const auto base_key = opencl_program_cache_key(base);

    auto different_artifact = base;
    different_artifact.artifact_key = "artifact://unit/opencl/eltwise/v2";
    EXPECT_NE(opencl_program_cache_key(different_artifact), base_key);

    auto different_abi = base;
    different_abi.abi_fingerprint = "abi://unit/opencl/eltwise/v2";
    EXPECT_NE(opencl_program_cache_key(different_abi), base_key);

    auto different_chunk = base;
    different_chunk.source_id = "opencl/generated/eltwise_f32/chunk0";
    different_chunk.entry_point = "gfx_eltwise_f32_chunk0";
    EXPECT_NE(opencl_program_cache_key(different_chunk), base_key);

    auto different_source = base;
    different_source.source += "\n";
    EXPECT_NE(opencl_program_cache_key(different_source), base_key);
}

TEST(GfxOpenClRuntimeBundleContractTest,
     ProgramCacheKeyRejectsIncompleteDescriptorIdentity) {
    auto request = make_contract_program_request();
    request.artifact_key.clear();
    EXPECT_THROW(opencl_program_cache_key(request), ov::Exception);

    request = make_contract_program_request();
    request.stage_record_key = 0;
    EXPECT_THROW(opencl_program_cache_key(request), ov::Exception);

    request = make_contract_program_request();
    request.backend_domain = "metal";
    EXPECT_THROW(opencl_program_cache_key(request), ov::Exception);
}

}  // namespace
}  // namespace gfx_plugin
}  // namespace ov
