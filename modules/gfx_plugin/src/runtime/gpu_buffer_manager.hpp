// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <optional>
#include <string>

#include "openvino/core/type/element_type.hpp"
#include "runtime/gpu_buffer.hpp"
#include "runtime/gpu_device_info.hpp"

namespace ov {
namespace gfx_plugin {

class GfxProfiler;
struct GpuBufferDesc;

// Backend-neutral buffer manager interface (backend implementations derive from
// this).
class GpuBufferManager {
public:
  virtual ~GpuBufferManager() = default;

  virtual std::optional<GpuExecutionDeviceInfo>
  query_execution_device_info() const {
    return std::nullopt;
  }

  virtual bool supports_const_cache() const { return false; }
  virtual GpuBuffer wrap_const(const std::string & /*key*/,
                               const void * /*data*/, size_t /*bytes*/,
                               ov::element::Type /*type*/) {
    return {};
  }
  virtual GpuBuffer allocate_temp(const GpuBufferDesc & /*desc*/) { return {}; }
  virtual void release_temp(GpuBuffer && /*buf*/) {}
  virtual void begin_const_upload_batch() {}
  virtual void
  flush_const_upload_batch(GpuCommandBufferHandle /*command_buffer*/,
                           GfxProfiler * /*profiler*/) {}
  virtual void end_const_upload_batch() {}
};

} // namespace gfx_plugin
} // namespace ov
