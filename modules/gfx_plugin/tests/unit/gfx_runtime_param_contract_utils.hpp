// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "unit/gfx_manifest_executable_contract_utils.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

class UnitMetadataBufferManager final : public GpuBufferManager {
public:
  GpuBuffer wrap_const(const std::string &, const void *, size_t bytes,
                       ov::element::Type type) override {
    GpuBuffer buffer;
    buffer.buffer = reinterpret_cast<GpuBufferHandle>(m_next_handle);
    buffer.size = bytes;
    buffer.type = type;
    buffer.persistent = true;
    buffer.allocation_uid = allocate_gpu_buffer_uid();
    m_next_handle += 0x1000u;
    return buffer;
  }

private:
  uintptr_t m_next_handle = 0x100000u;
};


} // namespace
} // namespace gfx_plugin
} // namespace ov
