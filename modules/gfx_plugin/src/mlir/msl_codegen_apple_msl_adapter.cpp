// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen_apple_msl.hpp"

#include <utility>
#include <vector>

#include "kernel_ir/gfx_custom_kernel_families.hpp"
#include "mlir/gfx_mlir_kernel_metadata.hpp"
#include "mlir/gfx_mpsrt_metadata.hpp"
#include "mlir/msl_codegen.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

bool is_apple_msl_custom_kernel_manifest(
    const GfxKernelStageManifest &manifest) {
  return manifest.valid &&
         manifest.backend_domain == GfxKernelBackendDomain::AppleMsl &&
         manifest.execution_kind == GfxKernelExecutionKind::CustomKernel &&
         manifest.custom_kernel.valid &&
         manifest.custom_kernel.external_buffer_abi.valid;
}

GfxMslRuntimeBindingPlan
make_binding_plan_from_manifest(const GfxKernelStageManifest &manifest,
                                std::vector<int32_t> scalar_args) {
  if (!is_apple_msl_custom_kernel_manifest(manifest)) {
    return {};
  }

  auto plan = make_msl_runtime_binding_plan_from_stage_manifest(manifest);
  if (!plan.valid() || plan.scalar_arg_count != scalar_args.size()) {
    return {};
  }
  plan.runtime_binding.scalar_args = std::move(scalar_args);
  plan.stage_manifest.custom_kernel.scalar_args =
      plan.runtime_binding.scalar_args;
  return plan;
}

GfxMslRuntimeBindingPlan
make_binding_plan_from_existing_manifest(mlir::ModuleOp module,
                                         std::vector<int32_t> scalar_args) {
  GfxKernelStageManifest manifest{};
  if (!module ||
      !detail::gfx_mpsrt_read_stage_manifest_attrs(module, manifest)) {
    return {};
  }
  return make_binding_plan_from_manifest(manifest, std::move(scalar_args));
}

GfxMslRuntimeBindingPlan make_binding_plan_for_adapter_request(
    mlir::ModuleOp module, std::string_view stage_type,
    std::string_view entry_point, std::vector<int32_t> scalar_args) {
  auto plan = make_binding_plan_from_existing_manifest(module, scalar_args);
  if (plan.valid()) {
    return plan;
  }

  const auto custom_kernel_plan =
      make_gfx_custom_kernel_stage_plan(stage_type, entry_point);
  if (!custom_kernel_plan.valid) {
    return {};
  }
  return make_binding_plan_from_manifest(custom_kernel_plan.stage_manifest,
                                         std::move(scalar_args));
}

} // namespace

bool annotate_apple_msl_custom_kernel_binding(
    mlir::ModuleOp module, std::string_view stage_type,
    std::string_view entry_point, std::vector<int32_t> scalar_args) {
  if (!module) {
    return false;
  }

  auto plan = make_binding_plan_for_adapter_request(
      module, stage_type, entry_point, std::move(scalar_args));
  return annotate_msl_module_with_runtime_binding_plan(module, plan);
}

void require_apple_msl_custom_kernel_binding(mlir::ModuleOp module,
                                             std::string_view stage_type,
                                             std::string_view entry_point,
                                             std::vector<int32_t> scalar_args) {
  OPENVINO_ASSERT(annotate_apple_msl_custom_kernel_binding(
                      module, stage_type, entry_point, std::move(scalar_args)),
                  "GFX Metal MSL: failed to derive runtime binding from stage "
                  "manifest for ",
                  stage_type, " / ", entry_point);
}

size_t infer_apple_msl_custom_kernel_arg_count(mlir::ModuleOp module,
                                               size_t fallback,
                                               std::string_view entry_point) {
  GfxKernelStageManifest manifest{};
  if (!module ||
      !detail::gfx_mpsrt_read_stage_manifest_attrs(module, manifest) ||
      !is_apple_msl_custom_kernel_manifest(manifest)) {
    return fallback;
  }
  return infer_kernel_arg_count_from_module(
      module, fallback, entry_point,
      /*allow_legacy_operand_attrs=*/false);
}

} // namespace gfx_plugin
} // namespace ov
