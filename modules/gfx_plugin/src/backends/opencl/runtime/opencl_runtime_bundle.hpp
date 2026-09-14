// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>
#include <vector>

namespace ov {
namespace gfx_plugin {

struct OpenClRuntimeBundleInfo {
    std::string library_path;
    std::string bundle_dir;
    std::string clspv_path;
    std::string llvm_spirv_path;
    bool plugin_adjacent = false;
};

std::vector<std::string> opencl_runtime_library_candidates(const std::string& module_dir);
OpenClRuntimeBundleInfo describe_opencl_runtime_bundle(const std::string& library_path,
                                                       const std::string& module_dir);
void configure_opencl_runtime_bundle_tools(const OpenClRuntimeBundleInfo& bundle);

}  // namespace gfx_plugin
}  // namespace ov
