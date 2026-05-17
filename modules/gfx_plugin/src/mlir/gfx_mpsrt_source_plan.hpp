// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "kernel_ir/gfx_codegen_backend.hpp"
#include "mlir/gfx_mlir_kernel_metadata.hpp"
#include "mlir/gfx_mpsrt_metadata.hpp"
#include "mlir/gfx_stage_kernel_binding.hpp"

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace ov {
namespace gfx_plugin {

enum class GfxMpsrtKernelSourcePlanKind {
  None,
  SingleStage,
  MultiStage,
};

struct GfxMpsrtKernelSourcePlan {
  GfxMpsrtKernelSourcePlanKind kind = GfxMpsrtKernelSourcePlanKind::None;
  KernelSource source;
  bool requires_mpsrt_model = false;
  std::string record_key;
  GfxMpsrtStageKind first_stage_kind = GfxMpsrtStageKind::Unknown;
  GfxMpsrtStageKind last_stage_kind = GfxMpsrtStageKind::Unknown;
  bool has_runtime_binding = false;
  KernelRuntimeBindingState runtime_binding;

  bool valid() const {
    return kind != GfxMpsrtKernelSourcePlanKind::None && source.module;
  }
};

namespace detail {

struct GfxMpsrtKernelSourceOptions {
  std::string msl_source;
  std::function<std::string(mlir::ModuleOp)> msl_generator;
  std::vector<uint32_t> spirv_binary;
  std::function<std::vector<uint32_t>(mlir::ModuleOp)> spirv_generator;

  bool has_source_payload() const {
    return !msl_source.empty() || static_cast<bool>(msl_generator) ||
           !spirv_binary.empty() || static_cast<bool>(spirv_generator);
  }
};

} // namespace detail

inline bool
gfx_mpsrt_stage_needs_custom_kernel_source(const GfxMpsrtStageDesc &stage) {
  return stage.kind == GfxMpsrtStageKind::MSLDispatch ||
         stage.kind == GfxMpsrtStageKind::SPIRVDispatch ||
         stage.stage_manifest.execution_kind ==
             GfxKernelExecutionKind::CustomKernel;
}

inline bool
gfx_mpsrt_stage_is_io_only_apple_mps_vendor(GfxMpsrtStageKind kind) {
  switch (kind) {
  case GfxMpsrtStageKind::MPSPool2D:
  case GfxMpsrtStageKind::MPSResize2D:
  case GfxMpsrtStageKind::MPSSoftmax:
  case GfxMpsrtStageKind::MPSTopK:
  case GfxMpsrtStageKind::MPSSdpa:
    return true;
  default:
    return false;
  }
}

inline bool gfx_mpsrt_source_plan_is_io_only_apple_mps_vendor(
    const GfxMpsrtKernelSourcePlan &plan) {
  return plan.requires_mpsrt_model &&
         plan.kind == GfxMpsrtKernelSourcePlanKind::SingleStage &&
         plan.first_stage_kind == plan.last_stage_kind &&
         gfx_mpsrt_stage_is_io_only_apple_mps_vendor(plan.first_stage_kind) &&
         plan.source.msl_source.empty() &&
         !static_cast<bool>(plan.source.msl_generator);
}

inline bool make_gfx_mpsrt_io_only_source_plan_runtime_binding(
    const GfxMpsrtBuilderPlan &builder_plan, KernelRuntimeBindingState &out) {
  out = {};
  if (!builder_plan.external_buffer_abi_valid ||
      builder_plan.external_buffer_roles.empty()) {
    return false;
  }

  uint32_t input_arg_count = 0;
  uint32_t output_arg_count = 0;
  for (const auto role : builder_plan.external_buffer_roles) {
    switch (role) {
    case GfxMpsrtExternalBufferRole::TensorInput:
    case GfxMpsrtExternalBufferRole::ConstBuffer:
    case GfxMpsrtExternalBufferRole::RuntimeParams:
    case GfxMpsrtExternalBufferRole::Metadata:
      ++input_arg_count;
      break;
    case GfxMpsrtExternalBufferRole::TensorOutput:
      ++output_arg_count;
      break;
    case GfxMpsrtExternalBufferRole::Unknown:
    default:
      return false;
    }
  }
  if (input_arg_count == 0 || output_arg_count == 0) {
    return false;
  }

  out = make_stage_compact_buffer_kernel_runtime_binding(input_arg_count);
  return true;
}

inline std::string gfx_mpsrt_stage_entry_point(const GfxMpsrtStageDesc &stage) {
  if (stage.stage_manifest.custom_kernel.valid &&
      !stage.stage_manifest.custom_kernel.entry_point.empty()) {
    return stage.stage_manifest.custom_kernel.entry_point;
  }
  if (!stage.kernel_name.empty()) {
    return stage.kernel_name;
  }
  return gfx_mpsrt_stage_kind_name(stage.kind);
}

namespace detail {

inline bool
gfx_mpsrt_source_stage_manifest_signature(const GfxMpsrtStageDesc &stage,
                                          uint32_t &arg_count,
                                          uint32_t &output_arg_count) {
  const auto &manifest = stage.stage_manifest;
  if (!manifest.valid ||
      manifest.execution_kind != GfxKernelExecutionKind::CustomKernel ||
      !manifest.custom_kernel.valid ||
      !manifest.custom_kernel.external_buffer_abi.valid) {
    return false;
  }

  const auto roles = materialize_kernel_external_buffer_roles(
      manifest.custom_kernel.external_buffer_abi);
  if (roles.empty()) {
    return false;
  }

  uint32_t resolved_arg_count = 0;
  uint32_t resolved_output_arg_count = 0;
  for (const auto role : roles) {
    if (is_gfx_kernel_buffer_role(role) || is_gfx_kernel_scalar_role(role)) {
      ++resolved_arg_count;
    }
    if (is_gfx_kernel_output_role(role)) {
      ++resolved_output_arg_count;
    }
  }
  if (resolved_arg_count == 0 || resolved_output_arg_count == 0) {
    return false;
  }

  arg_count = resolved_arg_count;
  output_arg_count = resolved_output_arg_count;
  return true;
}

inline bool gfx_mpsrt_source_stage_requires_exact_manifest_signature(
    const GfxMpsrtStageDesc &stage) {
  return gfx_mpsrt_stage_needs_custom_kernel_source(stage);
}

inline uint32_t
gfx_mpsrt_source_plan_arg_count(const GfxMpsrtModuleBuilderPlan &module_plan,
                                const GfxMpsrtStageDesc &source_stage,
                                const GfxMpsrtKernelSourceOptions &options) {
  uint32_t manifest_arg_count = 0;
  uint32_t manifest_output_arg_count = 0;
  if (gfx_mpsrt_source_stage_manifest_signature(
          source_stage, manifest_arg_count, manifest_output_arg_count)) {
    return manifest_arg_count;
  }
  if (module_plan.builder_plan.external_buffer_count != 0) {
    return module_plan.builder_plan.external_buffer_count;
  }
  (void)options;
  return static_cast<uint32_t>(module_plan.builder_plan.input_values.size() +
                               module_plan.builder_plan.output_values.size());
}

inline uint32_t gfx_mpsrt_source_plan_output_arg_count(
    const GfxMpsrtModuleBuilderPlan &module_plan,
    const GfxMpsrtStageDesc &source_stage,
    const GfxMpsrtKernelSourceOptions &options) {
  uint32_t manifest_arg_count = 0;
  uint32_t manifest_output_arg_count = 0;
  if (gfx_mpsrt_source_stage_manifest_signature(
          source_stage, manifest_arg_count, manifest_output_arg_count)) {
    return manifest_output_arg_count;
  }
  if (module_plan.builder_plan.external_output_buffer_count != 0) {
    return module_plan.builder_plan.external_output_buffer_count;
  }
  (void)options;
  return static_cast<uint32_t>(module_plan.builder_plan.output_values.size());
}

inline GfxMpsrtKernelSourcePlan
make_mpsrt_kernel_source_plan_from_module(mlir::ModuleOp module,
                                          GfxMpsrtKernelSourceOptions options) {
  GfxMpsrtKernelSourcePlan plan{};
  if (!module) {
    return plan;
  }

  GfxMpsrtModuleBuilderPlan module_plan;
  if (!build_module_mpsrt_builder_plan(module, module_plan)) {
    return plan;
  }

  const auto &program = module_plan.program;
  if (!program.valid || program.stages.empty()) {
    return plan;
  }

  const GfxMpsrtStageDesc *first_stage = nullptr;
  const GfxMpsrtStageDesc *last_stage = nullptr;
  const GfxMpsrtStageDesc *source_stage = nullptr;
  if (program.multi_stage) {
    first_stage = &program.stages.front().stage;
    last_stage = &program.stages.back().stage;
    for (auto it = program.stages.rbegin(); it != program.stages.rend(); ++it) {
      if (gfx_mpsrt_stage_needs_custom_kernel_source(it->stage)) {
        source_stage = &it->stage;
        break;
      }
    }
    if (!source_stage) {
      source_stage = last_stage;
    }
    plan.kind = GfxMpsrtKernelSourcePlanKind::MultiStage;
    plan.record_key = program.record_key;
  } else {
    first_stage = &module_plan.stage_plan.stage;
    last_stage = first_stage;
    source_stage = first_stage;
    plan.kind = GfxMpsrtKernelSourcePlanKind::SingleStage;
    plan.record_key = gfx_mpsrt_stage_plan_record_key(module_plan.stage_plan);
  }

  if (!first_stage || !last_stage || !source_stage) {
    return {};
  }

  uint32_t manifest_arg_count = 0;
  uint32_t manifest_output_arg_count = 0;
  const bool has_manifest_signature = gfx_mpsrt_source_stage_manifest_signature(
      *source_stage, manifest_arg_count, manifest_output_arg_count);
  const bool source_stage_needs_custom_source =
      gfx_mpsrt_source_stage_requires_exact_manifest_signature(*source_stage);
  if (source_stage_needs_custom_source) {
    if (!has_manifest_signature || !options.has_source_payload()) {
      return {};
    }
  } else if (options.has_source_payload()) {
    return {};
  }

  plan.first_stage_kind = first_stage->kind;
  plan.last_stage_kind = last_stage->kind;
  plan.requires_mpsrt_model = true;
  plan.source.module = module;
  plan.source.entry_point = gfx_mpsrt_stage_entry_point(*source_stage);
  plan.source.msl_source = std::move(options.msl_source);
  plan.source.msl_generator = std::move(options.msl_generator);
  plan.source.spirv_binary = std::move(options.spirv_binary);
  plan.source.spirv_generator = std::move(options.spirv_generator);
  if (has_manifest_signature) {
    plan.source.signature.arg_count = manifest_arg_count;
    plan.source.signature.output_arg_count = manifest_output_arg_count;
  } else {
    plan.source.signature.arg_count =
        gfx_mpsrt_source_plan_arg_count(module_plan, *source_stage, options);
    plan.source.signature.output_arg_count =
        gfx_mpsrt_source_plan_output_arg_count(module_plan, *source_stage,
                                               options);
  }
  if (gfx_mpsrt_source_plan_is_io_only_apple_mps_vendor(plan)) {
    if (!make_gfx_mpsrt_io_only_source_plan_runtime_binding(
            module_plan.builder_plan, plan.runtime_binding)) {
      return {};
    }
    plan.has_runtime_binding = true;
  }
  return plan;
}

} // namespace detail

inline GfxMpsrtKernelSourcePlan
make_mpsrt_kernel_source_plan_from_module(mlir::ModuleOp module) {
  return detail::make_mpsrt_kernel_source_plan_from_module(
      module, detail::GfxMpsrtKernelSourceOptions{});
}

inline GfxMpsrtKernelSourcePlan
make_mpsrt_kernel_source_plan_from_msl_source(mlir::ModuleOp module,
                                              std::string msl_source) {
  detail::GfxMpsrtKernelSourceOptions options{};
  options.msl_source = std::move(msl_source);
  return detail::make_mpsrt_kernel_source_plan_from_module(module,
                                                           std::move(options));
}

inline GfxMpsrtKernelSourcePlan
make_mpsrt_kernel_source_plan_from_msl_generator(
    mlir::ModuleOp module,
    std::function<std::string(mlir::ModuleOp)> msl_generator) {
  detail::GfxMpsrtKernelSourceOptions options{};
  options.msl_generator = std::move(msl_generator);
  return detail::make_mpsrt_kernel_source_plan_from_module(module,
                                                           std::move(options));
}

inline GfxMpsrtKernelSourcePlan
make_mpsrt_kernel_source_plan_from_configured_source(KernelSource source) {
  detail::GfxMpsrtKernelSourceOptions options{};
  options.msl_source = std::move(source.msl_source);
  options.msl_generator = std::move(source.msl_generator);
  options.spirv_binary = std::move(source.spirv_binary);
  options.spirv_generator = std::move(source.spirv_generator);
  return detail::make_mpsrt_kernel_source_plan_from_module(source.module,
                                                           std::move(options));
}

} // namespace gfx_plugin
} // namespace ov
