// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "openvino/core/node.hpp"
#include "runtime/gfx_stage_runtime_values.hpp"

namespace ov {
namespace gfx_plugin {

RuntimeValuePlan plan_reshape_runtime_values(const RuntimeInputResolver &inputs,
                                             const ov::Node &node,
                                             std::string_view stage_name);

RuntimeValuePlan
plan_squeeze_unsqueeze_runtime_values(const RuntimeInputResolver &inputs,
                                      const ov::Node &node,
                                      std::string_view stage_name);

RuntimeValuePlan
plan_shape_preserving_runtime_values(const RuntimeInputResolver &inputs,
                                     const ov::Node &node,
                                     std::string_view stage_name);

RuntimeValuePlan plan_broadcast_runtime_values(
    const RuntimeInputResolver &inputs, const ov::Node &node,
    const ov::Shape &input_shape, std::string_view stage_name);

RuntimeValuePlan plan_convert_runtime_values(const RuntimeInputResolver &inputs,
                                             const ov::Node *node,
                                             std::string_view stage_name);

RuntimeSelectPlan plan_select_runtime_values(const RuntimeInputResolver &inputs,
                                             const ov::Node &node,
                                             std::string_view stage_name);

std::optional<RuntimeReduceInfo>
get_runtime_reduce_info(const std::shared_ptr<const ov::Node> &node);

RuntimeConcatPlan plan_concat_runtime_values(const RuntimeInputResolver &inputs,
                                             const ov::Node &node,
                                             std::string_view stage_name);

RuntimeGatherPlan plan_gather_runtime_values(const RuntimeInputResolver &inputs,
                                             const ov::Node &node,
                                             std::string_view stage_name);

RuntimeScatterUpdatePlan
plan_scatter_update_runtime_values(const RuntimeInputResolver &inputs,
                                   const ov::Node &node,
                                   std::string_view stage_name);

RuntimeSplitPlan plan_split_runtime_values(const ov::Node *node,
                                           const ov::Shape &input_shape,
                                           size_t output_count,
                                           std::string_view stage_name);

RuntimeTransposePlan
plan_transpose_runtime_values(const RuntimeInputResolver &inputs,
                              const ov::Node &node,
                              std::string_view stage_name);

RuntimeInterpolatePlan plan_interpolate_runtime_values(
    const RuntimeInputResolver &inputs, const std::vector<GpuTensor *> &outputs,
    const ov::Node &node, std::string_view stage_name);

RuntimeSoftmaxPlan
plan_softmax_runtime_values(const RuntimeInputResolver &inputs,
                            const ov::Node &node, std::string_view stage_name);

bool bind_small_i64_const_stage_outputs(
    GpuBufferManager *buffer_manager, const std::vector<GpuTensor *> &outputs,
    std::vector<GpuTensor> &cache, const std::shared_ptr<const ov::Node> &node,
    GfxProfiler *profiler, bool profiling_enabled, std::string_view stage_name,
    std::string_view suffix);

} // namespace gfx_plugin
} // namespace ov
