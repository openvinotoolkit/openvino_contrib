// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>
#include <utility>

namespace ov {
namespace gfx_plugin {

inline std::string& current_vulkan_pipeline_cache_dir_storage() {
    static thread_local std::string value;
    return value;
}

inline const std::string& current_vulkan_pipeline_cache_dir() {
    return current_vulkan_pipeline_cache_dir_storage();
}

class ScopedVulkanPipelineCacheDir final {
public:
    explicit ScopedVulkanPipelineCacheDir(std::string cache_dir)
        : m_prev(std::exchange(current_vulkan_pipeline_cache_dir_storage(), std::move(cache_dir))) {}

    ~ScopedVulkanPipelineCacheDir() {
        current_vulkan_pipeline_cache_dir_storage() = std::move(m_prev);
    }

    ScopedVulkanPipelineCacheDir(const ScopedVulkanPipelineCacheDir&) = delete;
    ScopedVulkanPipelineCacheDir& operator=(const ScopedVulkanPipelineCacheDir&) = delete;

private:
    std::string m_prev;
};

}  // namespace gfx_plugin
}  // namespace ov
