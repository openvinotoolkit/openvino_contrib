// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "kernel_ir/gfx_codegen_backend.hpp"

namespace ov {
namespace gfx_plugin {

void require_apple_msl_generated_kernel_source_binding(
    KernelSource &source, std::string_view stage_type,
    std::string_view entry_point, std::vector<int32_t> scalar_args = {});

} // namespace gfx_plugin
} // namespace ov
