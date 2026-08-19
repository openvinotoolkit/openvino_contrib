// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "backends/opencl/runtime/opencl_api.hpp"
#include "runtime/gpu_buffer_manager.hpp"
#include "runtime/immutable_gpu_buffer_cache.hpp"

namespace ov {
namespace gfx_plugin {

class OpenClBufferManager final : public GpuBufferManager {
public:
    explicit OpenClBufferManager(std::shared_ptr<OpenClRuntimeContext> context);
    ~OpenClBufferManager() override;

    std::optional<GpuExecutionDeviceInfo> query_execution_device_info() const override;
    bool supports_const_cache() const override { return true; }
    GpuBuffer wrap_const(const std::string& key,
                         const void* data,
                         size_t bytes,
                         ov::element::Type type) override;
    GpuBuffer allocate_temp(const GpuBufferDesc& desc) override;
    void release_temp(GpuBuffer&& buf) override;
    void flush_const_upload_batch(GpuCommandBufferHandle command_buffer,
                                  GfxProfiler* profiler) override;

private:
    GpuBuffer allocate_buffer(size_t bytes,
                              ov::element::Type type,
                              BufferUsage usage,
                              bool host_visible,
                              const char* label);

    std::shared_ptr<OpenClRuntimeContext> m_context;
    std::shared_ptr<ImmutableGpuBufferCache> m_const_cache;
};

}  // namespace gfx_plugin
}  // namespace ov
