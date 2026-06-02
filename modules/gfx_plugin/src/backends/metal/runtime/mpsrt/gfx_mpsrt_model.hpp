// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "kernel_ir/gfx_kernel_manifest.hpp"
#include "backends/metal/runtime/mpsrt/gfx_mpsrt_abi.hpp"
#include "backends/metal/runtime/mpsrt/gfx_mpsrt_builder_plan.hpp"
#include "backends/metal/runtime/mpsrt/gfx_mpsrt_plan.hpp"

namespace ov {
namespace gfx_plugin {
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
  std::vector<GfxKernelBufferRole> kernel_argument_roles;
  GfxMpsrtConv2DAbiDesc conv2d_desc{};
  GfxMpsrtGemmAbiDesc gemm_desc{};
  GfxMpsrtPool2DAbiDesc pool2d_desc{};
  GfxMpsrtResize2DAbiDesc resize2d_desc{};
  GfxMpsrtSoftmaxAbiDesc softmax_desc{};
  GfxMpsrtTopKAbiDesc topk_desc{};
  GfxMpsrtSdpaAbiDesc sdpa_desc{};
  std::vector<GfxMpsrtValue> inputs;
  std::vector<GfxMpsrtValue> outputs;
  std::vector<GfxMpsrtValue> kernel_buffer_order;
  std::vector<GfxMpsrtTensorAbiDesc> output_descs;
};

enum class MpsrtRuntimeResourceLifetime : uint32_t {
  Unknown = 0,
  External = 1,
  Model = 2,
  Transient = 3,
};

struct MpsrtRuntimeResource {
  uint32_t resource_index = 0;
  GfxMpsrtExternalBufferRole role = GfxMpsrtExternalBufferRole::Unknown;
  MpsrtRuntimeResourceLifetime lifetime = MpsrtRuntimeResourceLifetime::Unknown;
  uint32_t arg_index = 0;
  bool has_tensor_value = false;
  GfxMpsrtValue value = 0;
  GfxMpsrtTensorAbiDesc tensor_desc{};
};

struct MpsrtExternalBufferBinding {
  uint32_t arg_index = 0;
  uint32_t resource_index = 0;
};

struct MpsrtTensorBindingPlanEntry {
  uint32_t resource_index = 0;
  MpsrtRuntimeResourceLifetime lifetime = MpsrtRuntimeResourceLifetime::Unknown;
  uint32_t arg_index = 0;
  GfxMpsrtExternalBufferRole role = GfxMpsrtExternalBufferRole::Unknown;
  bool has_tensor_value = false;
  GfxMpsrtValue value = 0;
  GfxMpsrtTensorAbiDesc tensor_desc{};
  GfxMpsrtStorageBridgeDirection bridge_direction =
      GfxMpsrtStorageBridgeDirection::Unknown;
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
  std::vector<MpsrtRuntimeResource> resources;
  std::vector<MpsrtExternalBufferBinding> external_buffer_bindings;
  std::vector<GfxMpsrtStorageBridgeDesc> storage_bridges;
};

const MpsrtRuntimeResource *
find_mpsrt_external_resource(const MpsrtModel &model,
                             const MpsrtExternalBufferBinding &binding);

const MpsrtRuntimeResource *
find_mpsrt_resource_for_value(const MpsrtModel &model, GfxMpsrtValue value);

const MpsrtRuntimeTensor *find_mpsrt_tensor(const MpsrtModel &model,
                                            GfxMpsrtValue value);

const GfxMpsrtStorageBridgeDesc *
find_mpsrt_storage_bridge(const MpsrtModel &model, GfxMpsrtValue value);

bool mpsrt_value_list_contains(const std::vector<GfxMpsrtValue> &values,
                               GfxMpsrtValue value);

size_t mpsrt_model_external_buffer_abi_count(const MpsrtModel &model);

size_t
mpsrt_model_resource_lifetime_count(const MpsrtModel &model,
                                    MpsrtRuntimeResourceLifetime lifetime);

GfxMpsrtStorageBridgeDirection mpsrt_model_external_bridge_direction_for_value(
    const MpsrtModel &model, GfxMpsrtValue value,
    GfxMpsrtStorageBridgeDirection fallback_direction);

bool mpsrt_model_tensor_binding_plan(
    const MpsrtModel &model, std::vector<MpsrtTensorBindingPlanEntry> &plan,
    std::string *error = nullptr);

bool finalize_mpsrt_model_resources(MpsrtModel &model,
                                    std::string *error = nullptr);

MpsrtRuntimeStage make_mpsrt_runtime_stage_from_desc(
    const GfxMpsrtStageDesc &desc, const std::vector<GfxMpsrtValue> &inputs,
    const std::vector<GfxMpsrtValue> &outputs,
    const std::vector<GfxMpsrtTensorAbiDesc> &output_descs);

bool build_mpsrt_model_from_builder_plan(const GfxMpsrtBuilderPlan &plan,
                                         MpsrtModel &model,
                                         std::string *error = nullptr);

MpsrtModel
build_mpsrt_model_from_builder_plan_or_throw(const GfxMpsrtBuilderPlan &plan);

bool adapt_mpsrt_model_to_external_buffer_abi(MpsrtModel &model,
                                              uint32_t arg_count,
                                              uint32_t output_arg_count,
                                              std::string *error = nullptr);

} // namespace mpsrt
} // namespace gfx_plugin
} // namespace ov
