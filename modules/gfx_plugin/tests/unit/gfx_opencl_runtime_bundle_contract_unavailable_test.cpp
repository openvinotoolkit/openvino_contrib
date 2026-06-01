// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "plugin/gfx_backend_config.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

void expect_opencl_runtime_bundle_contract_unavailable() {
    EXPECT_FALSE(kGfxBackendOpenCLAvailable)
        << "This adapter is linked only when the OpenCL backend module is not "
           "present in the current build target.";
}

TEST(GfxOpenClRuntimeBundleContractTest,
     CandidateOrderPrefersPluginAdjacentOpenclBundle) {
    expect_opencl_runtime_bundle_contract_unavailable();
}

TEST(GfxOpenClRuntimeBundleContractTest,
     BundleDescriptionMarksPluginAdjacentOpenclBundle) {
    expect_opencl_runtime_bundle_contract_unavailable();
}

TEST(GfxOpenClRuntimeBundleContractTest,
     SystemLibraryDoesNotAdvertiseBundledToolPaths) {
    expect_opencl_runtime_bundle_contract_unavailable();
}

}  // namespace
}  // namespace gfx_plugin
}  // namespace ov
