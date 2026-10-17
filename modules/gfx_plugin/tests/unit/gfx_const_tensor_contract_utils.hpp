// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "unit/gfx_manifest_executable_contract_utils.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

class UnitDescriptorConstBufferManager final : public GpuBufferManager {
public:
  struct Upload {
    std::string key;
    std::vector<uint8_t> bytes;
    ov::element::Type type;
    uint64_t allocation_uid = 0;
  };

  bool supports_const_cache() const override { return true; }

  GpuBuffer wrap_const(const std::string &key, const void *data, size_t bytes,
                       ov::element::Type type) override {
    Upload upload;
    upload.key = key;
    upload.bytes.resize(bytes);
    if (bytes != 0) {
      std::memcpy(upload.bytes.data(), data, bytes);
    }
    upload.type = type;

    GpuBuffer buffer;
    buffer.buffer = reinterpret_cast<GpuBufferHandle>(m_next_handle);
    buffer.size = bytes;
    buffer.type = type;
    buffer.persistent = true;
    buffer.allocation_uid = allocate_gpu_buffer_uid();
    upload.allocation_uid = buffer.allocation_uid;
    m_next_handle += 0x1000u;
    uploads.push_back(std::move(upload));
    return buffer;
  }

  std::vector<Upload> uploads;

private:
  uintptr_t m_next_handle = 0x200000u;
};


} // namespace
} // namespace gfx_plugin
} // namespace ov
