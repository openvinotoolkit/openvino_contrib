// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/runtime/opencl_runtime_bundle.hpp"

#include <cstdlib>
#include <filesystem>

namespace ov {
namespace gfx_plugin {
namespace {

std::string normalize_path(const std::string& path) {
    if (path.empty()) {
        return {};
    }
    return std::filesystem::path(path).lexically_normal().generic_string();
}

std::string join_path(const std::string& dir, const std::string& leaf) {
    if (dir.empty()) {
        return leaf;
    }
    return normalize_path((std::filesystem::path(dir) / leaf).generic_string());
}

bool path_exists(const std::string& path) {
    std::error_code ec;
    return !path.empty() && std::filesystem::exists(std::filesystem::path(path), ec);
}

bool same_path_text(const std::string& lhs, const std::string& rhs) {
    return normalize_path(lhs) == normalize_path(rhs);
}

bool is_plugin_adjacent_bundle_dir(const std::string& bundle_dir, const std::string& module_dir) {
    if (module_dir.empty() || bundle_dir.empty()) {
        return false;
    }
    return same_path_text(bundle_dir, join_path(module_dir, "opencl")) ||
           same_path_text(bundle_dir, join_path(module_dir, "clvk")) ||
           same_path_text(bundle_dir, module_dir);
}

void set_env_to_existing_path(const char* name, const std::string& path) {
    if (std::getenv(name) != nullptr || !path_exists(path)) {
        return;
    }
    setenv(name, path.c_str(), 0);
}

}  // namespace

std::vector<std::string> opencl_runtime_library_candidates(const std::string& module_dir) {
    std::vector<std::string> candidates;
    const auto normalized_module_dir = normalize_path(module_dir);
    if (!normalized_module_dir.empty()) {
        const char* bundle_dirs[] = {"opencl", "clvk", ""};
        const char* library_names[] = {"libOpenCL.so", "libOpenCL.so.1", "libOpenCL.so.0.1"};
        for (const char* bundle_dir : bundle_dirs) {
            const auto base_dir =
                bundle_dir[0] == '\0' ? normalized_module_dir : join_path(normalized_module_dir, bundle_dir);
            for (const char* library_name : library_names) {
                candidates.push_back(join_path(base_dir, library_name));
            }
        }
    }

    candidates.push_back("libOpenCL.so");
    candidates.push_back("libOpenCL.so.1");
    candidates.push_back("/vendor/lib64/libOpenCL.so");
    candidates.push_back("/vendor/lib/libOpenCL.so");
    candidates.push_back("/system/vendor/lib64/libOpenCL.so");
    candidates.push_back("/system/vendor/lib/libOpenCL.so");
    return candidates;
}

OpenClRuntimeBundleInfo describe_opencl_runtime_bundle(const std::string& library_path,
                                                       const std::string& module_dir) {
    OpenClRuntimeBundleInfo info;
    info.library_path = normalize_path(library_path);
    if (info.library_path.empty()) {
        return info;
    }

    const auto parent = std::filesystem::path(info.library_path).parent_path();
    if (parent.empty()) {
        return info;
    }

    info.bundle_dir = normalize_path(parent.generic_string());
    info.plugin_adjacent = is_plugin_adjacent_bundle_dir(info.bundle_dir, normalize_path(module_dir));
    if (info.plugin_adjacent) {
        info.clspv_path = join_path(info.bundle_dir, "clspv");
        info.llvm_spirv_path = join_path(info.bundle_dir, "llvm-spirv");
    }
    return info;
}

void configure_opencl_runtime_bundle_tools(const OpenClRuntimeBundleInfo& bundle) {
    if (!bundle.plugin_adjacent) {
        return;
    }
    set_env_to_existing_path("CLVK_CLSPV_PATH", bundle.clspv_path);
    set_env_to_existing_path("CLVK_LLVMSPIRV_BIN", bundle.llvm_spirv_path);
}

}  // namespace gfx_plugin
}  // namespace ov
