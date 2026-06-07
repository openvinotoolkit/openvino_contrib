// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string_view>
#include <vector>

#include "runtime/executable_descriptor.hpp"
#include "runtime/gfx_stage_runtime_values.hpp"

namespace ov {
namespace gfx_plugin {

RuntimeValuePlan plan_reshape_runtime_values(const RuntimeInputResolver &inputs,
                                             const RuntimeStageExecutableDescriptor &descriptor,
                                             std::string_view stage_name);

RuntimeValuePlan
plan_squeeze_unsqueeze_runtime_values(const RuntimeInputResolver &inputs,
                                      const RuntimeStageExecutableDescriptor &descriptor,
                                      std::string_view stage_name);

RuntimeValuePlan
plan_shape_preserving_runtime_values(const RuntimeInputResolver &inputs,
                                     const RuntimeStageExecutableDescriptor &descriptor,
                                     std::string_view stage_name);

RuntimeValuePlan plan_convert_runtime_values(const RuntimeInputResolver &inputs,
                                             const RuntimeStageExecutableDescriptor &descriptor,
                                             std::string_view stage_name);

RuntimeGatherPlan plan_gather_runtime_values(const RuntimeInputResolver &inputs,
                                             const RuntimeStageExecutableDescriptor &descriptor,
                                             std::string_view stage_name);

RuntimeScatterUpdatePlan
plan_scatter_update_runtime_values(const RuntimeInputResolver &inputs,
                                   const RuntimeStageExecutableDescriptor &descriptor,
                                   std::string_view stage_name);

RuntimeSplitPlan plan_split_runtime_values(const RuntimeInputResolver &inputs,
                                           const RuntimeStageExecutableDescriptor &descriptor,
                                           size_t output_count,
                                           std::string_view stage_name);

RuntimeTransposePlan
plan_transpose_runtime_values(const RuntimeInputResolver &inputs,
                              const RuntimeStageExecutableDescriptor &descriptor,
                              std::string_view stage_name);

RuntimeInterpolatePlan plan_interpolate_runtime_values(
    const RuntimeInputResolver &inputs, const std::vector<GpuTensor *> &outputs,
    const RuntimeStageExecutableDescriptor &descriptor,
    std::string_view stage_name);

RuntimeSoftmaxPlan
plan_softmax_runtime_values(const RuntimeInputResolver &inputs,
                            const RuntimeStageExecutableDescriptor &descriptor,
                            std::string_view stage_name);

} // namespace gfx_plugin
} // namespace ov
