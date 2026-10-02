// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string_view>

#include "common/artifact_payload.hpp"
#include "compiler/cache_envelope.hpp"
#include "compiler/executable_bundle.hpp"

namespace ov {
namespace gfx_plugin {

struct GfxOpenClSourceArtifact;

namespace compiler {

::ov::gfx_plugin::KernelArtifactOrigin
classify_opencl_kernel_artifact_origin(std::string_view kernel_unit_id) noexcept;

KernelArtifactPayloadResolver make_opencl_kernel_artifact_payload_resolver();
KernelArtifactDescriptorResolver
make_opencl_kernel_artifact_descriptor_resolver();
CacheBackendPayloadEncoder make_opencl_cache_payload_encoder();
CacheBackendPayloadDecoder make_opencl_cache_payload_decoder();

bool finalize_opencl_kernel_artifact_descriptor_contract(
    KernelArtifactDescriptor &descriptor,
    const ::ov::gfx_plugin::GfxOpenClSourceArtifact &artifact);
bool opencl_source_artifact_matches_descriptor_contract(
    const KernelArtifactDescriptor &descriptor,
    const ::ov::gfx_plugin::GfxOpenClSourceArtifact &artifact);

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
