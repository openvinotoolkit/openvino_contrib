// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/gfx_backend_custom_kernel_adapter.hpp"

#include "openvino/core/except.hpp"

#include <string>
#include <utility>

namespace ov {
namespace gfx_plugin {
namespace {

bool is_backend_custom_kernel_manifest(const GfxKernelStageManifest &manifest) {
  return manifest.valid &&
         manifest.execution_kind == GfxKernelExecutionKind::CustomKernel &&
         manifest.custom_kernel.valid &&
         manifest.custom_kernel.external_buffer_abi.valid;
}

bool is_backend_custom_kernel_manifest_for_domain(
    const GfxKernelStageManifest &manifest,
    GfxKernelBackendDomain backend_domain) {
  return is_backend_custom_kernel_manifest(manifest) &&
         manifest.backend_domain == backend_domain;
}

void set_semantic_io_roles_from_external_roles(
    GfxKernelStageManifest &manifest,
    const std::vector<GfxKernelBufferRole> &roles) {
  manifest.semantic_input_roles.clear();
  manifest.semantic_output_roles.clear();
  for (const auto role : roles) {
    if (role == GfxKernelBufferRole::TensorInput ||
        role == GfxKernelBufferRole::ConstTensor) {
      manifest.semantic_input_roles.push_back(role);
    } else if (role == GfxKernelBufferRole::TensorOutput) {
      manifest.semantic_output_roles.push_back(role);
    }
  }
}

} // namespace

GfxKernelRuntimeBindingPlan
make_backend_custom_kernel_binding_plan_from_stage_manifest(
    const GfxKernelStageManifest &manifest) {
  return make_kernel_runtime_binding_plan_from_stage_manifest(manifest);
}

bool read_backend_custom_kernel_stage_manifest_from_module(
    mlir::ModuleOp module, GfxKernelBackendDomain backend_domain,
    GfxKernelStageManifest &manifest) {
  manifest = {};
  return module &&
         detail::gfx_mpsrt_read_stage_manifest_attrs(module, manifest) &&
         is_backend_custom_kernel_manifest_for_domain(manifest, backend_domain);
}

std::string_view backend_custom_kernel_specialization_prefix(
    GfxKernelBackendDomain backend_domain, GfxKernelStorageKind storage) {
  OPENVINO_ASSERT(storage == GfxKernelStorageKind::Buffer,
                  "GFX MLIR: custom-kernel specialization prefix is only "
                  "defined for buffer storage");
  switch (backend_domain) {
  case GfxKernelBackendDomain::AppleMsl:
    return "apple_msl:buffer:";
  case GfxKernelBackendDomain::OpenCl:
    return "opencl:buffer:";
  case GfxKernelBackendDomain::AppleMps:
  case GfxKernelBackendDomain::Unknown:
  default:
    OPENVINO_THROW("GFX MLIR: custom-kernel specialization prefix is not "
                   "defined for backend domain ",
                   gfx_kernel_backend_domain_name(backend_domain));
  }
}

GfxKernelRuntimeBindingPlan make_backend_custom_kernel_binding_plan(
    std::string_view stage_type, std::string_view entry_point,
    GfxKernelBackendDomain backend_domain, GfxKernelStorageKind storage,
    std::string_view specialization_prefix) {
  const auto custom_kernel_plan = make_gfx_custom_kernel_stage_plan(
      stage_type, entry_point, backend_domain, storage, specialization_prefix);
  if (!custom_kernel_plan.valid) {
    return {};
  }
  return make_backend_custom_kernel_binding_plan_from_stage_manifest(
      custom_kernel_plan.stage_manifest);
}

GfxKernelRuntimeBindingPlan make_backend_custom_kernel_binding_plan(
    std::string_view stage_type, std::string_view entry_point,
    std::vector<int32_t> scalar_args, GfxKernelBackendDomain backend_domain,
    GfxKernelStorageKind storage, std::string_view specialization_prefix) {
  auto plan = make_backend_custom_kernel_binding_plan(
      stage_type, entry_point, backend_domain, storage, specialization_prefix);
  if (!plan.valid || plan.scalar_arg_count != scalar_args.size()) {
    return {};
  }
  plan.runtime_binding.scalar_args = std::move(scalar_args);
  plan.stage_manifest.custom_kernel.scalar_args =
      plan.runtime_binding.scalar_args;
  return plan;
}

GfxKernelRuntimeBindingPlan
make_backend_custom_kernel_binding_plan(GfxKernelBackendDomain backend_domain,
                                        std::string_view stage_type,
                                        std::string_view entry_point) {
  constexpr auto storage = GfxKernelStorageKind::Buffer;
  return make_backend_custom_kernel_binding_plan(
      stage_type, entry_point, backend_domain, storage,
      backend_custom_kernel_specialization_prefix(backend_domain, storage));
}

GfxKernelRuntimeBindingPlan make_backend_custom_kernel_binding_plan(
    GfxKernelBackendDomain backend_domain, std::string_view stage_type,
    std::string_view entry_point, std::vector<int32_t> scalar_args) {
  constexpr auto storage = GfxKernelStorageKind::Buffer;
  return make_backend_custom_kernel_binding_plan(
      stage_type, entry_point, std::move(scalar_args), backend_domain, storage,
      backend_custom_kernel_specialization_prefix(backend_domain, storage));
}

GfxCustomKernelStagePlan make_backend_custom_kernel_stage_plan_view(
    const GfxKernelRuntimeBindingPlan &binding) {
  GfxCustomKernelStagePlan plan{};
  plan.valid = binding.valid;
  plan.stage_manifest = binding.stage_manifest;
  return plan;
}

GfxKernelRuntimeBindingPlan make_backend_custom_kernel_direct_io_binding_plan(
    std::string_view stage_type, std::string_view entry_point,
    size_t tensor_input_count, size_t output_count,
    GfxKernelBackendDomain backend_domain, GfxKernelStorageKind storage,
    std::string_view specialization_prefix) {
  if (tensor_input_count == 0 || output_count == 0) {
    return {};
  }

  auto custom_kernel_plan = make_gfx_custom_kernel_stage_plan(
      stage_type, entry_point, backend_domain, storage, specialization_prefix);
  if (!custom_kernel_plan.valid || !custom_kernel_plan.stage_manifest.valid) {
    return {};
  }

  auto manifest = custom_kernel_plan.stage_manifest;
  manifest.custom_kernel.entry_point = std::string(entry_point);
  manifest.custom_kernel.external_buffer_abi =
      make_gfx_kernel_direct_io_abi(static_cast<uint32_t>(tensor_input_count),
                                    static_cast<uint32_t>(output_count));
  manifest.semantic_input_roles.assign(tensor_input_count,
                                       GfxKernelBufferRole::TensorInput);
  manifest.semantic_output_roles.assign(output_count,
                                        GfxKernelBufferRole::TensorOutput);
  auto plan =
      make_backend_custom_kernel_binding_plan_from_stage_manifest(manifest);
  return plan;
}

GfxKernelRuntimeBindingPlan make_backend_custom_kernel_roles_binding_plan(
    std::string_view stage_type, std::string_view entry_point,
    std::vector<GfxKernelBufferRole> roles,
    GfxKernelBackendDomain backend_domain, GfxKernelStorageKind storage,
    std::string_view specialization_prefix) {
  if (roles.empty()) {
    return {};
  }

  auto custom_kernel_plan = make_gfx_custom_kernel_stage_plan(
      stage_type, entry_point, backend_domain, storage, specialization_prefix);
  if (!custom_kernel_plan.valid || !custom_kernel_plan.stage_manifest.valid) {
    return {};
  }

  auto manifest = custom_kernel_plan.stage_manifest;
  manifest.custom_kernel.entry_point = std::string(entry_point);
  set_semantic_io_roles_from_external_roles(manifest, roles);
  manifest.custom_kernel.external_buffer_abi =
      make_gfx_kernel_roles_abi(std::move(roles));
  return make_backend_custom_kernel_binding_plan_from_stage_manifest(manifest);
}

GfxKernelRuntimeBindingPlan
make_backend_custom_kernel_binding_plan_from_module_or_request(
    mlir::ModuleOp module, std::string_view stage_type,
    std::string_view entry_point, std::vector<int32_t> scalar_args,
    GfxKernelBackendDomain backend_domain, GfxKernelStorageKind storage,
    std::string_view specialization_prefix) {
  GfxKernelStageManifest manifest{};
  if (read_backend_custom_kernel_stage_manifest_from_module(
          module, backend_domain, manifest)) {
    auto plan =
        make_backend_custom_kernel_binding_plan_from_stage_manifest(manifest);
    if (plan.valid) {
      const bool can_override_manifest =
          manifest.backend_domain != GfxKernelBackendDomain::AppleMsl;
      const bool requested_entry =
          !entry_point.empty() &&
          manifest.custom_kernel.entry_point != entry_point;
      const bool requested_scalars =
          !scalar_args.empty() && plan.scalar_arg_count != scalar_args.size();
      if (requested_scalars || (can_override_manifest && requested_entry)) {
        return make_backend_custom_kernel_binding_plan(
            stage_type, entry_point, std::move(scalar_args), backend_domain,
            storage, specialization_prefix);
      }
      if (plan.scalar_arg_count == 0 || !scalar_args.empty()) {
        if (plan.scalar_arg_count != scalar_args.size()) {
          return {};
        }
        plan.runtime_binding.scalar_args = std::move(scalar_args);
        plan.stage_manifest.custom_kernel.scalar_args =
            plan.runtime_binding.scalar_args;
      }
      return plan;
    }
  }

  if (scalar_args.empty()) {
    return make_backend_custom_kernel_binding_plan(stage_type, entry_point,
                                                   backend_domain, storage,
                                                   specialization_prefix);
  }
  return make_backend_custom_kernel_binding_plan(
      stage_type, entry_point, std::move(scalar_args), backend_domain, storage,
      specialization_prefix);
}

GfxKernelRuntimeBindingPlan make_backend_custom_kernel_source_binding_plan(
    const KernelSource &source, GfxKernelBackendDomain backend_domain,
    std::string_view stage_type, std::string_view entry_point,
    std::vector<int32_t> scalar_args) {
  const std::string resolved_entry =
      entry_point.empty() ? source.entry_point : std::string(entry_point);
  constexpr auto storage = GfxKernelStorageKind::Buffer;
  return make_backend_custom_kernel_binding_plan_from_module_or_request(
      source.module, stage_type, resolved_entry, std::move(scalar_args),
      backend_domain, storage,
      backend_custom_kernel_specialization_prefix(backend_domain, storage));
}

bool annotate_backend_custom_kernel_module_with_binding_plan(
    mlir::ModuleOp module, const GfxKernelRuntimeBindingPlan &plan) {
  if (!module || !plan.valid) {
    return false;
  }
  detail::gfx_mpsrt_set_stage_manifest_attrs(module, plan.stage_manifest);
  module->removeAttr("gfx.kernel_operand_kinds");
  module->removeAttr("gfx.kernel_operand_arg_indices");
  module->removeAttr("gfx.kernel_scalar_values");
  return true;
}

bool configure_backend_custom_kernel_source_from_binding_plan(
    KernelSource &source, const GfxKernelRuntimeBindingPlan &plan) {
  if (!plan.valid) {
    return false;
  }
  if (source.module && !annotate_backend_custom_kernel_module_with_binding_plan(
                           source.module, plan)) {
    return false;
  }
  return configure_backend_custom_kernel_source_signature(source, plan);
}

bool configure_backend_custom_kernel_source_binding(
    KernelSource &source, GfxKernelBackendDomain backend_domain,
    std::string_view stage_type, std::string_view entry_point,
    std::vector<int32_t> scalar_args) {
  auto plan = make_backend_custom_kernel_source_binding_plan(
      source, backend_domain, stage_type, entry_point,
      std::move(scalar_args));
  return configure_backend_custom_kernel_source_from_binding_plan(source, plan);
}

void require_backend_custom_kernel_source_binding(
    KernelSource &source, GfxKernelBackendDomain backend_domain,
    std::string_view stage_type, std::string_view entry_point,
    std::vector<int32_t> scalar_args) {
  OPENVINO_ASSERT(
      configure_backend_custom_kernel_source_binding(source, backend_domain,
                                                     stage_type, entry_point,
                                                     std::move(scalar_args)),
      "GFX MLIR: failed to configure custom-kernel source binding for ",
      stage_type, " / ", entry_point);
}

GfxKernelRuntimeBindingPlan require_backend_custom_kernel_binding_plan(
    GfxKernelBackendDomain backend_domain, std::string_view stage_type,
    std::string_view entry_point, const std::vector<int32_t> &scalar_args,
    std::string_view stage_name) {
  auto plan = make_backend_custom_kernel_binding_plan(
      backend_domain, stage_type, entry_point, scalar_args);
  OPENVINO_ASSERT(plan.valid, "GFX MLIR: ", stage_type, " / ", entry_point,
                  " custom-kernel runtime binding manifest is invalid for "
                  "stage ",
                  stage_name);
  return plan;
}

KernelRuntimeBindingState require_backend_custom_kernel_runtime_binding(
    GfxKernelBackendDomain backend_domain, std::string_view stage_type,
    std::string_view entry_point, const std::vector<int32_t> &scalar_args,
    std::string_view stage_name) {
  return require_backend_custom_kernel_binding_plan(backend_domain,
                                                    stage_type, entry_point,
                                                    scalar_args, stage_name)
      .runtime_binding;
}

GfxKernelRuntimeBindingPlan annotate_required_backend_custom_kernel_binding(
    mlir::ModuleOp module, GfxKernelBackendDomain backend_domain,
    std::string_view stage_type, std::string_view entry_point,
    const std::vector<int32_t> &scalar_args, std::string_view stage_name) {
  auto plan = require_backend_custom_kernel_binding_plan(
      backend_domain, stage_type, entry_point, scalar_args, stage_name);
  annotate_backend_custom_kernel_module_with_binding_plan(module, plan);
  return plan;
}

GfxKernelRuntimeBindingPlan annotate_required_backend_custom_kernel_abi_binding(
    mlir::ModuleOp module, GfxKernelBackendDomain backend_domain,
    std::string_view stage_type, std::string_view entry_point,
    std::string_view stage_name) {
  auto plan = make_backend_custom_kernel_binding_plan(backend_domain,
                                                      stage_type, entry_point);
  OPENVINO_ASSERT(plan.valid, "GFX MLIR: ", stage_type, " / ", entry_point,
                  " custom-kernel ABI binding manifest is invalid for stage ",
                  stage_name);
  OPENVINO_ASSERT(
      annotate_backend_custom_kernel_module_with_binding_plan(module, plan),
      "GFX MLIR: ", stage_type, " / ", entry_point,
      " failed to annotate custom-kernel ABI binding for stage ", stage_name);
  return plan;
}

GfxKernelRuntimeBindingPlan
annotate_required_backend_custom_kernel_direct_io_binding(
    mlir::ModuleOp module, GfxKernelBackendDomain backend_domain,
    std::string_view stage_type, std::string_view entry_point,
    size_t tensor_input_count, size_t output_count, std::string_view stage_name) {
  constexpr auto storage = GfxKernelStorageKind::Buffer;
  auto plan = make_backend_custom_kernel_direct_io_binding_plan(
      stage_type, entry_point, tensor_input_count, output_count,
      backend_domain, storage,
      backend_custom_kernel_specialization_prefix(backend_domain, storage));
  OPENVINO_ASSERT(plan.valid, "GFX MLIR: ", stage_type, " / ", entry_point,
                  " direct-IO custom-kernel runtime binding manifest is "
                  "invalid for stage ",
                  stage_name);
  OPENVINO_ASSERT(
      annotate_backend_custom_kernel_module_with_binding_plan(module, plan),
      "GFX MLIR: ", stage_type, " / ", entry_point,
      " failed to annotate direct-IO custom-kernel runtime binding for stage ",
      stage_name);
  return plan;
}

bool configure_backend_custom_kernel_source_signature(
    KernelSource &source, const GfxKernelStageManifest &manifest) {
  if (!is_backend_custom_kernel_manifest(manifest)) {
    return false;
  }

  const auto roles = materialize_gfx_kernel_external_buffer_roles(
      manifest.custom_kernel.external_buffer_abi);
  if (roles.empty()) {
    return false;
  }

  uint32_t arg_count = 0;
  uint32_t output_arg_count = 0;
  for (const auto role : roles) {
    if (is_gfx_kernel_buffer_role(role) || is_gfx_kernel_scalar_role(role)) {
      ++arg_count;
    }
    if (is_gfx_kernel_output_role(role)) {
      ++output_arg_count;
    }
  }
  if (arg_count == 0 || output_arg_count == 0) {
    return false;
  }

  source.signature.arg_count = arg_count;
  source.signature.output_arg_count = output_arg_count;
  return true;
}

bool configure_backend_custom_kernel_source_signature_from_module(
    KernelSource &source) {
  GfxKernelStageManifest manifest{};
  if (!source.module ||
      !detail::gfx_mpsrt_read_stage_manifest_attrs(source.module, manifest)) {
    return false;
  }
  return configure_backend_custom_kernel_source_signature(source, manifest);
}

bool configure_backend_custom_kernel_source_signature(
    KernelSource &source, const GfxKernelRuntimeBindingPlan &plan) {
  return plan.valid && configure_backend_custom_kernel_source_signature(
                           source, plan.stage_manifest);
}

size_t infer_backend_custom_kernel_arg_count(
    mlir::ModuleOp module, GfxKernelBackendDomain backend_domain,
    size_t fallback, std::string_view entry_point) {
  GfxKernelStageManifest manifest{};
  if (!read_backend_custom_kernel_stage_manifest_from_module(
          module, backend_domain, manifest)) {
    return fallback;
  }
  return infer_kernel_arg_count_from_module(module, fallback, entry_point,
                                            backend_domain);
}

size_t require_backend_manifest_arg_count(
    mlir::ModuleOp module, GfxKernelBackendDomain backend_domain,
    std::string_view entry_point, std::string_view stage_name) {
  OPENVINO_ASSERT(module, "GFX MLIR: ", stage_name,
                  " custom-kernel ABI requires an MLIR module");

  size_t arg_count = 0;
  if (infer_kernel_arg_count_from_mpsrt_typed_program_manifest(
          module, arg_count, entry_point, backend_domain) ||
      infer_kernel_arg_count_from_stage_manifest(module, arg_count, entry_point,
                                                 backend_domain)) {
    return arg_count;
  }

  OPENVINO_THROW("GFX MLIR: ", stage_name,
                 " is missing compiler-owned custom-kernel manifest ABI for ",
                 backend_domain == GfxKernelBackendDomain::OpenCl ? "OpenCL"
                                                                  : "Apple MSL",
                 " entry ", entry_point);
}

} // namespace gfx_plugin
} // namespace ov
