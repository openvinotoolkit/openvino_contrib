// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen.hpp"

#include "mlir/gfx_apple_stage_pipeline.hpp"
#include "mlir/gfx_mlir_kernel_builder.hpp"
#include "mlir/gfx_mpsrt_metadata.hpp"
#include "mlir/mlir_kernel_plan_utils.hpp"
#include "mlir/msl_codegen_apple_mps.hpp"
#include "mlir/msl_codegen_apple_msl.hpp"
#include "runtime/gfx_shape_utils.hpp"
#include "transforms/mlir_fused_ops.hpp"

#include "mlir/IR/Builders.h"

#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/depth_to_space.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/reverse.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/space_to_depth.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"
#include "ov_ops/rms.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <utility>

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

} // namespace

void force_apple_msl_buffer_placement(GfxStageOptimizationPlan &plan,
                                      std::string_view stage_type) {
  plan.placement.domain = GfxStageBackendDomain::AppleMsl;
  plan.placement.storage = GfxStageStorageKind::Buffer;
  plan.placement.uses_vendor_primitive = false;
  plan.placement.uses_custom_kernel = true;
  plan.placement.specialization_key =
      std::string("apple_msl:buffer:") + std::string(stage_type);
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

  lowering_plan.custom_kernel_plan = make_gfx_custom_kernel_stage_plan(
      stage_type, lowering_plan.stage_plan.stage.dispatch_entry_point);
  if (!lowering_plan.custom_kernel_plan.valid) {
    return {};
  }
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

void configure_msl_kernel_source_for_plan(KernelSource &source,
                                          std::string_view stage_type) {
  if (!source.module) {
    return;
  }

  GfxMpsrtModuleStagePlan stage_plan;
  if (!read_module_mpsrt_stage_plan(source.module, stage_plan) ||
      stage_plan.stage.kind != GfxMpsrtStageKind::MSLDispatch) {
    return;
  }

  auto custom_kernel_plan =
      make_gfx_custom_kernel_stage_plan(stage_type, source.entry_point);
  if (!custom_kernel_plan.valid) {
    custom_kernel_plan = make_gfx_custom_kernel_stage_plan(
        stage_plan.stage.stage_type, source.entry_point);
  }
  const auto &custom_kernel = custom_kernel_plan.stage_manifest.custom_kernel;
  if (!custom_kernel_plan.valid || !custom_kernel.valid ||
      custom_kernel.entry_point.empty()) {
    return;
  }

  const std::string source_entry = source.entry_point.empty()
                                       ? stage_plan.stage.kernel_name
                                       : source.entry_point;
  const std::string required_entry = custom_kernel.entry_point;
  if (!source.msl_source.empty()) {
    source.msl_source = normalize_msl_source_for_kernel_plan(
        std::move(source.msl_source), source_entry, custom_kernel_plan);
  }
  if (source.msl_generator) {
    auto generator = std::move(source.msl_generator);
    source.msl_generator = [generator = std::move(generator), source_entry,
                            custom_kernel_plan](mlir::ModuleOp module) mutable {
      return normalize_msl_source_for_kernel_plan(
          generator(module), source_entry, custom_kernel_plan);
    };
  }
  source.entry_point = required_entry;
  GfxMpsrtExternalBufferAbiPlan external_buffer_abi{};
  if (gfx_mpsrt_external_buffer_abi_from_kernel_manifest(
          source.module, external_buffer_abi, source.signature.arg_count,
          source.signature.output_arg_count) &&
      read_module_mpsrt_stage_plan(source.module, stage_plan)) {
    GfxAppleMslStageLoweringPlan lowering_plan{};
    lowering_plan.valid = true;
    lowering_plan.stage_plan = std::move(stage_plan);
    lowering_plan.custom_kernel_plan = std::move(custom_kernel_plan);
    (void)materialize_apple_msl_typed_program(source.module, lowering_plan,
                                              external_buffer_abi);
  }
}

GfxMpsrtKernelSourcePlan
configure_msl_kernel_source_plan(KernelSource source,
                                 std::string_view stage_type) {
  configure_msl_kernel_source_for_plan(source, stage_type);
  return make_mpsrt_kernel_source_plan_from_configured_source(
      std::move(source));
}

GfxMpsrtKernelSourcePlan configure_msl_kernel_source_plan_for_node(
    KernelSource source, const std::shared_ptr<const ov::Node> &node,
    const GpuBufferManager *buffer_manager, std::string_view stage_type,
    bool has_bias, bool has_activation, bool has_batchnorm) {
  if (!source.module || !node) {
    return {};
  }

  const auto msl_kernel_plan =
      make_gfx_custom_kernel_stage_plan(stage_type, source.entry_point);
  if (!msl_kernel_plan.valid) {
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
  auto source_plan = configure_msl_kernel_source_plan(source, stage_type);
  if (source_plan.valid()) {
    return source_plan;
  }
  configure_msl_kernel_source_for_plan(source, stage_type);
  return make_mpsrt_kernel_source_plan_from_configured_source(
      std::move(source));
}

GfxMpsrtKernelSourcePlan configure_apple_metal_kernel_source_plan_for_stage(
    KernelSource &source, const std::shared_ptr<const ov::Node> &node,
    const GpuBufferManager *buffer_manager, std::string_view stage_type,
    bool has_bias, bool has_activation, bool has_batchnorm,
    ActivationKind activation, const ov::element::Type &storage_type,
    bool has_runtime_slice_params,
    const std::optional<ov::Shape> &runtime_input_shape) {
  if (source.module) {
    auto vendor_source_plan =
        configure_apple_mps_vendor_kernel_source_plan_for_node(
            source, node, buffer_manager, stage_type, has_bias, has_activation,
            has_batchnorm, activation);
    if (vendor_source_plan.valid()) {
      return vendor_source_plan;
    }
  }

  configure_apple_metal_msl_kernel_source(
      source, node, stage_type, storage_type, has_runtime_slice_params,
      runtime_input_shape);
  if (!source.module) {
    return {};
  }
  return configure_msl_kernel_source_plan_for_node(
      source, node, buffer_manager, stage_type, has_bias, has_activation,
      has_batchnorm);
}

GfxMpsrtKernelSourcePlan configure_msl_kernel_source_plan_for_spec(
    KernelSource source, const KernelSpec &spec,
    const GpuBufferManager *buffer_manager, std::string_view entry_point) {
  if (source.entry_point.empty()) {
    source.entry_point = std::string(entry_point);
  }
  return configure_msl_kernel_source_plan_for_node(
      std::move(source), spec.node(), buffer_manager, spec.type(),
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false);
}

void annotate_msl_module_with_stage_plan(mlir::ModuleOp module,
                                         const GfxStageOptimizationPlan &plan,
                                         const std::string &stage_type,
                                         std::string_view kernel_entry_point) {
  GfxAppleStagePipelineOptions options{};
  options.plan = plan;
  options.stage_type = stage_type;
  options.kernel_entry_point = std::string(kernel_entry_point);
  (void)run_gfx_apple_stage_pipeline(module, options);
}

GfxMslRuntimeBindingPlan make_msl_runtime_binding_plan_from_stage_manifest(
    const GfxKernelStageManifest &manifest) {
  GfxMslRuntimeBindingPlan plan{};
  const auto runtime_plan =
      make_kernel_runtime_binding_plan_from_stage_manifest(manifest);
  if (!runtime_plan.valid) {
    return plan;
  }
  plan.stage_manifest = runtime_plan.stage_manifest;
  plan.runtime_binding = runtime_plan.runtime_binding;
  plan.scalar_arg_count = runtime_plan.scalar_arg_count;
  return plan;
}

GfxMslRuntimeBindingPlan
make_msl_runtime_binding_plan_for_custom_kernel(std::string_view stage_type,
                                                std::string_view entry_point) {
  const auto custom_kernel_plan =
      make_gfx_custom_kernel_stage_plan(stage_type, entry_point);
  if (!custom_kernel_plan.valid) {
    return {};
  }
  return make_msl_runtime_binding_plan_from_stage_manifest(
      custom_kernel_plan.stage_manifest);
}

GfxMslRuntimeBindingPlan make_msl_runtime_binding_plan_for_custom_kernel(
    std::string_view stage_type, std::string_view entry_point,
    std::vector<int32_t> scalar_args) {
  auto plan =
      make_msl_runtime_binding_plan_for_custom_kernel(stage_type, entry_point);
  if (!plan.valid() || plan.scalar_arg_count != scalar_args.size()) {
    return {};
  }
  plan.runtime_binding.scalar_args = std::move(scalar_args);
  plan.stage_manifest.custom_kernel.scalar_args =
      plan.runtime_binding.scalar_args;
  return plan;
}

GfxMslRuntimeBindingPlan
make_msl_runtime_binding_plan_for_direct_io_custom_kernel(
    std::string_view stage_type, std::string_view entry_point,
    size_t tensor_input_count, size_t output_count) {
  if (tensor_input_count == 0 || output_count == 0) {
    return {};
  }

  auto custom_kernel_plan =
      make_gfx_custom_kernel_stage_plan(stage_type, entry_point);
  if (!custom_kernel_plan.valid || !custom_kernel_plan.stage_manifest.valid) {
    return {};
  }

  auto manifest = custom_kernel_plan.stage_manifest;
  manifest.custom_kernel.external_buffer_abi =
      make_gfx_kernel_direct_io_abi(static_cast<uint32_t>(tensor_input_count),
                                    static_cast<uint32_t>(output_count));
  return make_msl_runtime_binding_plan_from_stage_manifest(manifest);
}

GfxDirectSplitMslKernelSourcePlan make_direct_split_msl_kernel_source_plan(
    std::string_view stage_type, const ov::element::Type &element_type,
    const ov::Shape &input_shape, const std::vector<size_t> &split_sizes,
    uint32_t axis_len, uint32_t inner_stride, mlir::ModuleOp module) {
  GfxDirectSplitMslKernelSourcePlan plan{};
  if (input_shape.empty() || split_sizes.empty() || axis_len == 0 ||
      inner_stride == 0) {
    return plan;
  }

  const auto binding =
      make_msl_runtime_binding_plan_for_direct_io_custom_kernel(
          stage_type, "split_kernel", 1, split_sizes.size());
  if (!binding.valid()) {
    return plan;
  }
  mlir::ModuleOp manifest_module;
  if (module) {
    manifest_module =
        mlir::ModuleOp::create(mlir::UnknownLoc::get(module.getContext()));
    OPENVINO_ASSERT(
        annotate_msl_module_with_runtime_binding_plan(manifest_module, binding),
        "GFX MSL: failed to annotate direct Split stage manifest");
  }

  const auto total_elems = ov::shape_size(input_shape);
  const auto scalar = msl_type_from_element(element_type);
  std::ostringstream msl;
  msl << "#include <metal_stdlib>\nusing namespace metal;\n";
  msl << "constant uint OFFSETS[" << (split_sizes.size() + 1) << "] = {0";
  uint64_t prefix = 0;
  for (auto sz : split_sizes) {
    prefix += static_cast<uint64_t>(sz);
    msl << ", " << prefix;
  }
  msl << "};\n";
  msl << "constant uint AXIS_DIM = " << axis_len << ";\n";
  msl << "constant uint STRIDE_AFTER = " << inner_stride << ";\n";
  msl << "constant uint OUTER_STRIDE = AXIS_DIM * STRIDE_AFTER;\n";
  msl << "kernel void split_kernel(\n";
  msl << "  device const " << scalar << "* input [[buffer(0)]],\n";
  for (size_t oi = 0; oi < split_sizes.size(); ++oi) {
    msl << "  device " << scalar << "* out" << oi << " [[buffer(" << (oi + 1)
        << ")]],\n";
  }
  msl << "  uint gid [[thread_position_in_grid]]) {\n";
  msl << "    uint total = " << static_cast<uint32_t>(total_elems) << ";\n";
  msl << "    if (gid >= total) return;\n";
  msl << "    uint axis_idx = (gid / STRIDE_AFTER) % AXIS_DIM;\n";
  msl << "    uint outer = gid / OUTER_STRIDE;\n";
  msl << "    uint inner = gid % STRIDE_AFTER;\n";
  msl << "    uint o = 0;\n";
  msl << "    while (o + 1 < " << (split_sizes.size() + 1)
      << " && axis_idx >= OFFSETS[o + 1]) ++o;\n";
  msl << "    uint local_axis = axis_idx - OFFSETS[o];\n";
  msl << "    uint dst_axis_extent = OFFSETS[o + 1] - OFFSETS[o];\n";
  msl << "    uint dst_idx = (outer * dst_axis_extent + local_axis) * "
         "STRIDE_AFTER + inner;\n";
  msl << "    switch (o) {\n";
  for (size_t oi = 0; oi < split_sizes.size(); ++oi) {
    msl << "      case " << oi << ": out" << oi
        << "[dst_idx] = input[gid]; break;\n";
  }
  msl << "      default: break;\n";
  msl << "    }\n";
  msl << "}\n";

  plan.source =
      make_kernel_source(manifest_module, "split_kernel", msl.str(),
                         static_cast<uint32_t>(1 + split_sizes.size()));
  plan.source.signature.output_arg_count =
      static_cast<uint32_t>(split_sizes.size());
  plan.binding = binding;
  return plan;
}

bool annotate_msl_module_with_runtime_binding_plan(
    mlir::ModuleOp module, const GfxMslRuntimeBindingPlan &plan) {
  if (!module || !plan.valid()) {
    return false;
  }
  detail::gfx_mpsrt_set_stage_manifest_attrs(module, plan.stage_manifest);
  module->removeAttr("gfx.kernel_operand_kinds");
  module->removeAttr("gfx.kernel_operand_arg_indices");
  module->removeAttr("gfx.kernel_scalar_values");
  return true;
}

} // namespace gfx_plugin
} // namespace ov
