// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "compiler/executable_bundle.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

KernelArtifactPayloadResolver make_opencl_kernel_artifact_payload_resolver();

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
