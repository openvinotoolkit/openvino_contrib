// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>
#include <vector>

#include "runtime/gfx_mpsrt_abi.hpp"
#include "runtime/gfx_mpsrt_builder_plan.hpp"
#include "runtime/gfx_mpsrt_plan.hpp"

namespace ov {
namespace gfx_plugin {
namespace metal {
namespace mpsrt {

struct MpsrtRuntimeTensor {
    GfxMpsrtValue value = 0;
    GfxMpsrtTensorAbiDesc desc{};
};

struct MpsrtRuntimeStage {
    GfxMpsrtStageKind kind = GfxMpsrtStageKind::Unknown;
    std::string stage_record_key;
    std::string kernel_name;
    std::string dispatch_kernel_family;
    std::string dispatch_entry_point;
    uint32_t dispatch_kernel_family_id = 0;
    uint32_t dispatch_flags = GfxMpsrtMslDispatchFlagNone;
    uint32_t dispatch_threads_per_threadgroup = 0;
    bool dispatch_precompiled_kernel_required = false;
    GfxMpsrtMslDispatchAbiDesc msl_dispatch_desc{};
    GfxMpsrtConv2DAbiDesc conv2d_desc{};
    GfxMpsrtGemmAbiDesc gemm_desc{};
    GfxMpsrtPool2DAbiDesc pool2d_desc{};
    GfxMpsrtSoftmaxAbiDesc softmax_desc{};
    GfxMpsrtTopKAbiDesc topk_desc{};
    std::vector<GfxMpsrtValue> inputs;
    std::vector<GfxMpsrtValue> outputs;
    std::vector<GfxMpsrtValue> kernel_buffer_order;
    std::vector<GfxMpsrtTensorAbiDesc> output_descs;
};

struct MpsrtModel {
    std::string stage_record_key;
    std::vector<MpsrtRuntimeTensor> tensors;
    std::vector<MpsrtRuntimeStage> stages;
    std::vector<GfxMpsrtValue> semantic_input_values;
    std::vector<GfxMpsrtValue> semantic_output_values;
    std::vector<GfxMpsrtValue> input_values;
    std::vector<GfxMpsrtValue> output_values;
    std::vector<GfxMpsrtValue> external_values;
    std::vector<GfxMpsrtValue> external_input_values;
    std::vector<GfxMpsrtValue> external_output_values;
    std::vector<GfxMpsrtExternalBufferRole> external_buffer_roles;
    std::vector<GfxMpsrtStorageBridgeDesc> storage_bridges;
};

MpsrtRuntimeStage make_mpsrt_runtime_stage_from_desc(const GfxMpsrtStageDesc& desc,
                                                     const std::string& stage_record_key,
                                                     const std::vector<GfxMpsrtValue>& inputs,
                                                     const std::vector<GfxMpsrtValue>& outputs,
                                                     const std::vector<GfxMpsrtTensorAbiDesc>& output_descs);

bool build_mpsrt_model_from_builder_plan(const GfxMpsrtBuilderPlan& plan,
                                         MpsrtModel& model,
                                         std::string* error = nullptr);

MpsrtModel build_mpsrt_model_from_builder_plan_or_throw(const GfxMpsrtBuilderPlan& plan);

bool adapt_mpsrt_model_to_external_buffer_abi(MpsrtModel& model,
                                              uint32_t arg_count,
                                              uint32_t output_arg_count,
                                              std::string* error = nullptr);

}  // namespace mpsrt
}  // namespace metal
}  // namespace gfx_plugin
}  // namespace ov
