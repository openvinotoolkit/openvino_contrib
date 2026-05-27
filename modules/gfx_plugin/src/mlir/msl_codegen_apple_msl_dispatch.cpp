// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen_apple_msl_dispatch.hpp"
#include "mlir/msl_codegen_apple_msl.hpp"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "kernel_ir/gfx_custom_kernel_families.hpp"
#include "mlir/gfx_apple_stage_pipeline.hpp"
#include "mlir/gfx_backend_custom_kernel_adapter.hpp"
#include "mlir/gfx_mpsrt_metadata.hpp"
#include "mlir/msl_codegen_apple_msl_ops.hpp"
#include "runtime/gfx_stage_policy.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

bool is_msl_ident_char(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') || c == '_';
}

bool replace_kernel_entry_name(std::string &source,
                               std::string_view current_entry_point,
                               std::string_view required_entry_point) {
  if (current_entry_point.empty() || required_entry_point.empty() ||
      current_entry_point == required_entry_point) {
    return false;
  }

  const std::string needle = "kernel void " + std::string(current_entry_point);
  size_t pos = source.find(needle);
  while (pos != std::string::npos) {
    const size_t name_pos = pos + std::string("kernel void ").size();
    const size_t after_name = name_pos + current_entry_point.size();
    if (after_name < source.size() && !is_msl_ident_char(source[after_name])) {
      source.replace(name_pos, current_entry_point.size(),
                     required_entry_point);
      return true;
    }
    pos = source.find(needle, pos + 1);
  }
  return false;
}

GfxKernelStageFamily
resolve_apple_metal_msl_stage_family(const KernelSource &source,
                                     std::string_view stage_type) {
  GfxKernelStageManifest manifest{};
  if (source.module &&
      detail::gfx_mpsrt_read_stage_manifest_attrs(source.module, manifest) &&
      manifest.valid &&
      manifest.backend_domain == GfxKernelBackendDomain::AppleMsl &&
      manifest.execution_kind == GfxKernelExecutionKind::CustomKernel) {
    return manifest.stage_family;
  }
  const auto kernel_family =
      classify_gfx_custom_kernel_family(stage_type, source.entry_point);
  return gfx_kernel_stage_family_from_kernel_family(kernel_family);
}

std::optional<KernelSource> make_apple_metal_msl_kernel_source_for_stage_type(
    KernelSource source, const std::shared_ptr<const ov::Node> &node,
    std::string_view stage_type,
    const std::optional<ov::Shape> &runtime_input_shape) {
  auto apply_data_movement_source = [&]() {
    return make_apple_metal_data_movement_kernel_source(source, node);
  };
  auto apply_convolution_source = [&]() {
    return make_apple_metal_convolution_kernel_source(source, node);
  };
  auto apply_shape_source = [&]() {
    return make_apple_metal_shape_kernel_source(source, node);
  };
  auto apply_convert_source = [&]() {
    return make_apple_metal_convert_kernel_source(source, node);
  };
  auto apply_matmul_source = [&]() {
    return make_apple_metal_matmul_kernel_source(source, node);
  };
  auto apply_llm_source = [&]() {
    return make_apple_metal_llm_kernel_source(source, node);
  };
  auto apply_pool2d_source = [&]() {
    return make_apple_metal_pool2d_kernel_source(source, node);
  };
  auto apply_softmax_source = [&]() {
    return make_apple_metal_softmax_kernel_source(source, node,
                                                  runtime_input_shape);
  };
  auto apply_elementwise_source = [&]() {
    return make_apple_metal_elementwise_kernel_source(source, node);
  };
  auto apply_layout_source = [&]() {
    return make_apple_metal_layout_kernel_source(source, node);
  };
  auto apply_concat_split_source = [&]() {
    return make_apple_metal_concat_split_kernel_source(source, node);
  };
  auto apply_reduction_source = [&]() {
    return make_apple_metal_reduction_kernel_source(source, node);
  };
  auto apply_topk_source = [&]() {
    return make_apple_metal_topk_kernel_source(source, node);
  };
  auto apply_unary_source = [&]() {
    return make_apple_metal_unary_kernel_source(source, node);
  };

  if (stage_type == "TopK" || source.entry_point == "topk_kernel") {
    return apply_topk_source();
  }
  if (stage_type == "Convert" || source.entry_point == "convert_kernel") {
    return apply_convert_source();
  }

  switch (resolve_apple_metal_msl_stage_family(source, stage_type)) {
  case GfxKernelStageFamily::Convolution:
  case GfxKernelStageFamily::Conv3D:
  case GfxKernelStageFamily::GroupConvolution:
    return apply_convolution_source();
  case GfxKernelStageFamily::Gemm:
    return apply_matmul_source();
  case GfxKernelStageFamily::RmsnormRope:
    return apply_llm_source();
  case GfxKernelStageFamily::Pooling:
    return apply_pool2d_source();
  case GfxKernelStageFamily::Softmax:
  case GfxKernelStageFamily::AttentionSoftmax:
    return apply_softmax_source();
  case GfxKernelStageFamily::Eltwise:
    if (auto configured = apply_elementwise_source()) {
      return configured;
    }
    if (auto configured = apply_unary_source()) {
      return configured;
    }
    return apply_shape_source();
  case GfxKernelStageFamily::Activation:
    return apply_unary_source();
  case GfxKernelStageFamily::Transpose:
    if (auto configured = apply_data_movement_source()) {
      return configured;
    }
    return apply_layout_source();
  case GfxKernelStageFamily::ConcatSplit:
    return apply_concat_split_source();
  case GfxKernelStageFamily::Reduction:
    return apply_reduction_source();
  case GfxKernelStageFamily::TopK:
    return apply_topk_source();
  case GfxKernelStageFamily::Convert:
    return apply_convert_source();
  case GfxKernelStageFamily::Layout:
    return apply_layout_source();
  case GfxKernelStageFamily::GatherScatter:
    if (auto configured = apply_data_movement_source()) {
      return configured;
    }
    return apply_shape_source();
  case GfxKernelStageFamily::Resize:
    return apply_data_movement_source();
  case GfxKernelStageFamily::KvCache:
  case GfxKernelStageFamily::Unknown:
  default:
    return std::nullopt;
  }
}

std::optional<KernelSource> make_apple_metal_msl_kernel_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node,
    std::string_view stage_type, const ov::element::Type &storage_type,
    bool has_runtime_slice_params,
    const std::optional<ov::Shape> &runtime_input_shape) {
  const bool prefer_family_specific_source =
      stage_type == "TopK" ||
      resolve_apple_metal_msl_stage_family(source, stage_type) ==
          GfxKernelStageFamily::ConcatSplit;
  if (prefer_family_specific_source ||
      (!source.msl_generator && source.msl_source.empty())) {
    if (auto configured = make_apple_metal_msl_kernel_source_for_stage_type(
            source, node, stage_type, runtime_input_shape)) {
      source = std::move(*configured);
    }
  }

  if (auto configured = make_apple_metal_slice_kernel_source(
          source, node, storage_type, has_runtime_slice_params)) {
    source = std::move(*configured);
  }
  if (!source.msl_generator && source.msl_source.empty()) {
    return std::nullopt;
  }
  return source;
}

bool has_non_materializable_apple_msl_external_abi(
    const GfxKernelStageManifest &manifest) {
  if (!manifest.valid ||
      manifest.backend_domain != GfxKernelBackendDomain::AppleMsl ||
      manifest.execution_kind != GfxKernelExecutionKind::CustomKernel ||
      !manifest.custom_kernel.valid ||
      !manifest.custom_kernel.external_buffer_abi.valid) {
    return false;
  }
  const auto roles = materialize_gfx_kernel_external_buffer_roles(
      manifest.custom_kernel.external_buffer_abi);
  return roles.empty();
}

std::string
normalize_msl_source_for_kernel_plan(std::string source,
                                     std::string_view current_entry_point,
                                     const GfxCustomKernelStagePlan &plan) {
  const auto &custom_kernel = plan.stage_manifest.custom_kernel;
  if (!plan.valid || !custom_kernel.valid ||
      custom_kernel.entry_point.empty()) {
    return source;
  }
  (void)replace_kernel_entry_name(source, current_entry_point,
                                  custom_kernel.entry_point);
  return source;
}

} // namespace

bool configure_msl_generated_custom_kernel_source(
    KernelSource &source, std::string_view stage_type,
    GfxKernelRuntimeBindingPlan &binding,
    GfxCustomKernelStagePlan *custom_kernel_plan) {
  binding = make_backend_custom_kernel_source_binding_plan(
      source, /*is_opencl_backend=*/false, stage_type);
  if (!binding.valid || !binding.stage_manifest.custom_kernel.valid ||
      binding.stage_manifest.custom_kernel.entry_point.empty()) {
    return false;
  }

  auto plan_view = make_backend_custom_kernel_stage_plan_view(binding);
  if (!source.msl_source.empty()) {
    source.msl_source = normalize_msl_source_for_kernel_plan(
        std::move(source.msl_source), source.entry_point, plan_view);
  }
  source.entry_point = binding.stage_manifest.custom_kernel.entry_point;
  if (!configure_backend_custom_kernel_source_from_binding_plan(source,
                                                                binding)) {
    return false;
  }
  if (custom_kernel_plan) {
    *custom_kernel_plan = std::move(plan_view);
  }
  return true;
}

GfxMslGeneratedKernelSourcePlan
make_msl_generated_custom_kernel_source_plan(KernelSource source,
                                             std::string_view stage_type) {
  GfxMslGeneratedKernelSourcePlan plan{};
  if (!configure_msl_generated_custom_kernel_source(source, stage_type,
                                                    plan.binding, nullptr)) {
    return {};
  }
  plan.source = std::move(source);
  return plan;
}

GfxMslGeneratedKernelSourcePlan make_msl_generated_custom_kernel_source_plan(
    KernelSource source, const GfxKernelRuntimeBindingPlan &binding) {
  GfxMslGeneratedKernelSourcePlan plan{};
  plan.binding = binding;
  plan.source = std::move(source);
  if (!configure_backend_custom_kernel_source_from_binding_plan(plan.source,
                                                                plan.binding)) {
    return {};
  }
  return plan;
}

void force_apple_msl_buffer_placement(GfxStageOptimizationPlan &plan,
                                      std::string_view /*stage_type*/) {
  plan.placement.domain = GfxStageBackendDomain::AppleMsl;
  plan.placement.storage = GfxStageStorageKind::Buffer;
  plan.placement.uses_vendor_primitive = false;
  plan.placement.uses_custom_kernel = true;
  plan.placement.specialization_key.clear();
}

GfxAppleMslStageLoweringPlan materialize_apple_msl_stage_manifest(
    mlir::ModuleOp module, const GfxStageOptimizationPlan &plan,
    const std::string &stage_type, std::string_view kernel_entry_point) {
  GfxAppleMslStageLoweringPlan lowering_plan{};
  GfxAppleStagePipelineOptions options{};
  options.plan = plan;
  options.stage_type = stage_type;
  options.kernel_entry_point = std::string(kernel_entry_point);
  options.materialize_typed_program = false;
  const auto pipeline_result = run_gfx_apple_stage_pipeline(module, options);
  if (!pipeline_result.valid) {
    return lowering_plan;
  }
  lowering_plan.stage_plan = pipeline_result.stage_plan;
  if (lowering_plan.stage_plan.stage.kind != GfxMpsrtStageKind::MSLDispatch) {
    return {};
  }

  auto custom_kernel_binding =
      make_backend_custom_kernel_binding_plan_from_stage_manifest(
          lowering_plan.stage_plan.stage.stage_manifest);
  if (!custom_kernel_binding.valid ||
      lowering_plan.stage_plan.stage.stage_manifest.backend_domain !=
          GfxKernelBackendDomain::AppleMsl) {
    return {};
  }
  lowering_plan.custom_kernel_plan =
      make_backend_custom_kernel_stage_plan_view(custom_kernel_binding);
  lowering_plan.valid = true;
  return lowering_plan;
}

bool materialize_apple_msl_typed_program(
    mlir::ModuleOp module, const GfxAppleMslStageLoweringPlan &lowering_plan,
    const GfxMpsrtExternalBufferAbiPlan &external_buffer_abi) {
  if (!module || !lowering_plan.valid) {
    return false;
  }
  return materialize_module_mpsrt_ops_from_stage_plan(
      module, lowering_plan.stage_plan, external_buffer_abi);
}

namespace {

struct AppleMslConfiguredSource {
  KernelSource source;
};

AppleMslConfiguredSource
make_apple_msl_configured_source_for_plan(KernelSource source,
                                          std::string_view stage_type) {
  AppleMslConfiguredSource configured{std::move(source)};
  auto &configured_source = configured.source;

  if (!configured_source.module) {
    return configured;
  }

  GfxMpsrtModuleStagePlan stage_plan;
  if (!read_module_mpsrt_stage_plan(configured_source.module, stage_plan) &&
      !build_mpsrt_stage_plan_from_manifest(configured_source.module,
                                            stage_plan)) {
    return configured;
  }
  GfxKernelStageManifest canonical_manifest{};
  if (detail::gfx_mpsrt_read_stage_manifest_attrs(configured_source.module,
                                                  canonical_manifest) &&
      canonical_manifest.valid &&
      canonical_manifest.backend_domain == GfxKernelBackendDomain::AppleMsl &&
      canonical_manifest.execution_kind == GfxKernelExecutionKind::CustomKernel) {
    stage_plan.stage.stage_manifest = std::move(canonical_manifest);
    detail::gfx_mpsrt_apply_stage_manifest_to_stage_desc(stage_plan.stage);
    stage_plan.valid = finalize_mpsrt_module_stage_plan(stage_plan);
  }
  if (stage_plan.stage.kind != GfxMpsrtStageKind::MSLDispatch) {
    return configured;
  }

  const std::string source_entry = configured_source.entry_point.empty()
                                       ? stage_plan.stage.kernel_name
                                       : configured_source.entry_point;
  auto custom_kernel_binding =
      make_backend_custom_kernel_binding_plan_from_stage_manifest(
          stage_plan.stage.stage_manifest);
  if (custom_kernel_binding.valid &&
      stage_plan.stage.stage_manifest.backend_domain !=
          GfxKernelBackendDomain::AppleMsl) {
    custom_kernel_binding = {};
  }
  if (!custom_kernel_binding.valid &&
      has_non_materializable_apple_msl_external_abi(
          stage_plan.stage.stage_manifest)) {
    return configured;
  }
  if (!custom_kernel_binding.valid) {
    custom_kernel_binding = make_backend_custom_kernel_source_binding_plan(
        configured_source, /*is_opencl_backend=*/false, stage_type,
        source_entry);
  }
  if (!custom_kernel_binding.valid) {
    custom_kernel_binding = make_backend_custom_kernel_binding_plan(
        /*is_opencl_backend=*/false, gfx_mpsrt_stage_type(stage_plan.stage),
        source_entry);
  }
  const auto &custom_kernel =
      custom_kernel_binding.stage_manifest.custom_kernel;
  if (!custom_kernel_binding.valid || !custom_kernel.valid ||
      custom_kernel.entry_point.empty()) {
    return configured;
  }

  const std::string required_entry = custom_kernel.entry_point;
  if (!configured_source.msl_source.empty()) {
    configured_source.msl_source = normalize_msl_source_for_kernel_plan(
        std::move(configured_source.msl_source), source_entry,
        make_backend_custom_kernel_stage_plan_view(custom_kernel_binding));
  }
  if (configured_source.msl_generator) {
    auto generator = std::move(configured_source.msl_generator);
    configured_source.msl_generator = [generator = std::move(generator),
                                       source_entry, custom_kernel_binding](
                                          mlir::ModuleOp module) mutable {
      return normalize_msl_source_for_kernel_plan(
          generator(module), source_entry,
          make_backend_custom_kernel_stage_plan_view(custom_kernel_binding));
    };
  }
  stage_plan.stage.stage_manifest = custom_kernel_binding.stage_manifest;
  detail::gfx_mpsrt_apply_stage_manifest_to_stage_desc(stage_plan.stage);
  stage_plan.valid = finalize_mpsrt_module_stage_plan(stage_plan);
  detail::gfx_mpsrt_set_stage_manifest_attrs(configured_source.module,
                                             stage_plan.stage.stage_manifest);
  configured_source.entry_point = required_entry;
  (void)configure_backend_custom_kernel_source_signature(configured_source,
                                                         custom_kernel_binding);
  GfxMpsrtExternalBufferAbiPlan external_buffer_abi{};
  (void)gfx_mpsrt_external_buffer_abi_from_kernel_manifest(
      configured_source.module, external_buffer_abi);
  if (external_buffer_abi.valid) {
    GfxAppleMslStageLoweringPlan lowering_plan{};
    lowering_plan.valid = true;
    lowering_plan.stage_plan = std::move(stage_plan);
    lowering_plan.custom_kernel_plan =
        make_backend_custom_kernel_stage_plan_view(custom_kernel_binding);
    (void)materialize_apple_msl_typed_program(
        configured_source.module, lowering_plan, external_buffer_abi);
  }
  return configured;
}

} // namespace

GfxMpsrtKernelSourcePlan
configure_msl_kernel_source_plan(KernelSource source,
                                 std::string_view stage_type) {
  auto configured =
      make_apple_msl_configured_source_for_plan(std::move(source), stage_type);
  return make_mpsrt_kernel_source_plan_from_configured_source(
      std::move(configured.source));
}

static GfxMpsrtKernelSourcePlan configure_msl_kernel_source_plan_for_node(
    KernelSource source, const std::shared_ptr<const ov::Node> &node,
    const GpuBufferManager *buffer_manager, std::string_view stage_type,
    bool has_bias, bool has_activation, bool has_batchnorm) {
  if (!source.module || !node) {
    return {};
  }

  GfxKernelStageManifest source_manifest{};
  auto msl_kernel_binding =
      read_backend_custom_kernel_stage_manifest_from_module(
          source.module, GfxKernelBackendDomain::AppleMsl, source_manifest)
          ? make_backend_custom_kernel_binding_plan_from_stage_manifest(
                source_manifest)
          : make_backend_custom_kernel_binding_plan(
                /*is_opencl_backend=*/false, stage_type, source.entry_point);
  if (!msl_kernel_binding.valid) {
    return {};
  }
  if (source.entry_point.empty() &&
      !msl_kernel_binding.stage_manifest.custom_kernel.entry_point.empty()) {
    source.entry_point =
        msl_kernel_binding.stage_manifest.custom_kernel.entry_point;
  }
  if (!configure_backend_custom_kernel_source_from_binding_plan(
          source, msl_kernel_binding)) {
    return {};
  }

  auto plan = select_stage_optimization_plan(
      buffer_manager, GpuBackend::Metal, std::string(stage_type), node,
      node->get_output_element_type(0), has_bias, has_activation, has_batchnorm,
      GfxStageRuntimeTraits{});
  if (plan.placement.domain != GfxStageBackendDomain::AppleMsl) {
    force_apple_msl_buffer_placement(plan, stage_type);
  }

  annotate_msl_module_with_stage_plan(
      source.module, plan, std::string(stage_type), source.entry_point);
  if (resolve_apple_metal_msl_stage_family(source, stage_type) ==
      GfxKernelStageFamily::ConcatSplit) {
    if (auto configured =
            make_apple_metal_concat_split_kernel_source(source, node)) {
      source = std::move(*configured);
    }
  }
  return configure_msl_kernel_source_plan(std::move(source), stage_type);
}

void annotate_msl_module_with_stage_plan(mlir::ModuleOp module,
                                         const GfxStageOptimizationPlan &plan,
                                         const std::string &stage_type,
                                         std::string_view kernel_entry_point) {
  GfxAppleStagePipelineOptions options{};
  options.plan = plan;
  options.stage_type = stage_type;
  options.kernel_entry_point = std::string(kernel_entry_point);
  options.materialize_typed_program = true;
  (void)run_gfx_apple_stage_pipeline(module, options);
}

GfxMpsrtKernelSourcePlan configure_apple_metal_msl_kernel_source_plan(
    KernelSource &source, const std::shared_ptr<const ov::Node> &node,
    const GpuBufferManager *buffer_manager, std::string_view stage_type,
    const ov::element::Type &storage_type, bool has_runtime_slice_params,
    const std::optional<ov::Shape> &runtime_input_shape, bool has_bias,
    bool has_activation, bool has_batchnorm) {
  auto configured = make_apple_metal_msl_kernel_source(
      std::move(source), node, stage_type, storage_type,
      has_runtime_slice_params, runtime_input_shape);
  if (!configured) {
    return {};
  }
  source = std::move(*configured);
  if (!source.module) {
    return {};
  }
  return configure_msl_kernel_source_plan_for_node(
      source, node, buffer_manager, stage_type, has_bias, has_activation,
      has_batchnorm);
}

} // namespace gfx_plugin
} // namespace ov
