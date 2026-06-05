// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string_view>

#include "common/artifact_payload.hpp"
#include "compiler/executable_bundle.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

::ov::gfx_plugin::KernelArtifactOrigin
classify_opencl_kernel_artifact_origin(std::string_view kernel_unit_id) noexcept;

KernelArtifactPayloadResolver make_opencl_kernel_artifact_payload_resolver();

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
