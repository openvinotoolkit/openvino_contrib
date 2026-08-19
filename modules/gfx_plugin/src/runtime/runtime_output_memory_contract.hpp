// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>

#include "runtime/infer_pipeline_state.hpp"

namespace ov {
namespace gfx_plugin {

bool runtime_output_uses_transient_arena(const InferStage& stage, size_t output_index);

bool apply_runtime_output_memory_contract(const InferStage& stage,
                                          size_t output_index,
                                          GpuBufferDesc& desc,
                                          GpuTensor& output,
                                          const char* error_prefix = "GFX");

}  // namespace gfx_plugin
}  // namespace ov
