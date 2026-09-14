// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>
#include <string_view>
#include <utility>

#include "backends/metal/common/mpsrt/gfx_mpsrt_vendor_contract.hpp"
#include "common/artifact_payload.hpp"

namespace ov {
namespace gfx_plugin {

class GfxMetalVendorPrimitiveArtifactPayload final
    : public KernelArtifactPayload {
public:
  GfxMetalVendorPrimitiveArtifactPayload(
      std::string kernel_id, std::string backend_domain,
      std::string entry_point, GfxAppleMpsVendorPrimitiveContract contract)
      : m_kernel_id(std::move(kernel_id)),
        m_backend_domain(std::move(backend_domain)),
        m_entry_point(std::move(entry_point)),
        m_contract(std::move(contract)) {}

  KernelArtifactPayloadKind payload_kind() const noexcept override {
    return KernelArtifactPayloadKind::VendorDescriptor;
  }

  std::string_view backend_domain() const noexcept override {
    return m_backend_domain;
  }

  std::string_view source_id() const noexcept override { return m_kernel_id; }

  std::string_view entry_point() const noexcept override {
    return m_entry_point;
  }

  bool valid() const noexcept override {
    return !m_kernel_id.empty() && m_backend_domain == "metal" &&
           !m_entry_point.empty() && m_contract.valid &&
           m_contract.descriptor.kind != GfxAppleMpsVendorPrimitiveKind::None;
  }

  const GfxAppleMpsVendorPrimitiveContract &contract() const noexcept {
    return m_contract;
  }

private:
  std::string m_kernel_id;
  std::string m_backend_domain;
  std::string m_entry_point;
  GfxAppleMpsVendorPrimitiveContract m_contract;
};

} // namespace gfx_plugin
} // namespace ov
