// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "runtime/gfx_profiler.hpp"
#include "runtime/gpu_buffer_pool.hpp"
#include "runtime/infer_pipeline.hpp"
#include "runtime/stateful_variable_state.hpp"

namespace ov {
namespace gfx_plugin {

struct BackendResources;

bool sync_stateful_variable_host(StatefulVariableTensorState& slot,
                                 const std::string& variable_id,
                                 const BackendResources& resources,
                                 GfxProfiler* profiler = nullptr);

bool execute_stateful_stage(StatefulVariableStateMap& variable_states,
                            InferStage& stage,
                            const std::vector<GpuTensor*>& resolved_inputs,
                            GpuBufferPool& pool,
                            GpuCommandBufferHandle command_buffer,
                            GfxProfiler* profiler = nullptr);

bool try_bind_direct_stateful_assign_output(StatefulVariableStateMap& variable_states,
                                            InferStage& stage,
                                            const std::vector<GpuTensor*>& resolved_inputs,
                                            GpuBufferPool& pool,
                                            GfxProfiler* profiler = nullptr);

using StatefulBackendStageExecutor =
    std::function<void(InferStage&, const std::vector<GpuTensor*>&, GpuCommandBufferHandle)>;

void execute_infer_stage_with_stateful_contract(
    StatefulVariableStateMap& variable_states,
    InferStage& stage,
    const std::vector<GpuTensor*>& resolved_inputs,
    GpuBufferPool& pool,
    GpuCommandBufferHandle command_buffer,
    GfxProfiler* profiler = nullptr,
    const StatefulBackendStageExecutor& backend_execute = {});

}  // namespace gfx_plugin
}  // namespace ov
