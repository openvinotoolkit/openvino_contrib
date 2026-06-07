// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string_view>

#include "backends/metal/common/mpsrt/gfx_mpsrt_vendor_artifact_payload.hpp"
#include "compiler/executable_bundle.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

using ::ov::gfx_plugin::GfxMetalVendorPrimitiveArtifactPayload;

KernelArtifactPayloadResolver make_metal_kernel_artifact_payload_resolver();
KernelArtifactDescriptorResolver make_metal_kernel_artifact_descriptor_resolver();
std::string_view metal_mpsgraph_sdpa_vendor_kernel_unit_id() noexcept;
PipelineVendorAttentionArtifactResolver
make_metal_vendor_attention_artifact_resolver();

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
