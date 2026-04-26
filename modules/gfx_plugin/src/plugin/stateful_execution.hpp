// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>
#include <vector>

#include "plugin/infer_pipeline.hpp"
#include "plugin/infer_request_state.hpp"
#include "runtime/gpu_buffer_pool.hpp"

namespace ov {
namespace gfx_plugin {

bool is_stateful_read_value(const std::shared_ptr<const ov::Node>& node);
bool is_stateful_assign(const std::shared_ptr<const ov::Node>& node);
std::string get_stateful_variable_id(const std::shared_ptr<const ov::Node>& node);

bool execute_stateful_stage(InferRequestState& state,
                            InferStage& stage,
                            const std::vector<GpuTensor*>& resolved_inputs,
                            GpuBufferPool& pool,
                            GpuCommandBufferHandle command_buffer);

}  // namespace gfx_plugin
}  // namespace ov
