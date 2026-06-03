// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>
#include <string_view>

#include "compiler/executable_bundle.hpp"
#include "backends/metal/compiler/apple_vendor_descriptors.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

class GfxMetalVendorPrimitiveArtifactPayload final
    : public KernelArtifactPayload {
public:
  GfxMetalVendorPrimitiveArtifactPayload(
      std::string kernel_id, std::string backend_domain,
      std::string entry_point, GfxAppleMpsVendorPrimitiveContract contract);

  KernelArtifactPayloadKind payload_kind() const noexcept override;
  std::string_view backend_domain() const noexcept override;
  std::string_view source_id() const noexcept override;
  std::string_view entry_point() const noexcept override;
  bool valid() const noexcept override;

  const GfxAppleMpsVendorPrimitiveContract &contract() const noexcept {
    return m_contract;
  }

private:
  std::string m_kernel_id;
  std::string m_backend_domain;
  std::string m_entry_point;
  GfxAppleMpsVendorPrimitiveContract m_contract;
};

KernelArtifactPayloadResolver make_metal_kernel_artifact_payload_resolver();
std::string_view metal_mpsgraph_sdpa_vendor_kernel_unit_id() noexcept;
PipelineVendorAttentionArtifactResolver
make_metal_vendor_attention_artifact_resolver();

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
