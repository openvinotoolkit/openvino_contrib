// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/compiler/apple_mlir_stage_hooks.hpp"

#include <string>

#include "backends/metal/compiler/apple_mpsrt_const_tensor_sources.hpp"
#include "backends/metal/compiler/apple_mpsrt_conv_metadata.hpp"
#include "backends/metal/compiler/apple_stage_pipeline.hpp"
#include "backends/metal/compiler/msl_codegen_apple_mps.hpp"
#include "backends/metal/compiler/msl_codegen_apple_msl_activation.hpp"
#include "backends/metal/compiler/msl_codegen_apple_msl_reduction.hpp"
#include "backends/metal/compiler/msl_codegen_apple_msl_shape.hpp"
#include "backends/metal/compiler/msl_codegen_apple_msl_slice_static.hpp"
#include "backends/metal/compiler/msl_codegen_apple_msl_split.hpp"
#include "backends/metal/compiler/msl_codegen_attention.hpp"
#include "backends/metal/compiler/msl_codegen_compressed_matmul.hpp"
#include "backends/metal/compiler/msl_codegen_matmul_metal.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/runtime/tensor.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

bool is_apple_conv_like_stage(std::string_view stage_type) {
  return stage_type == "Convolution" || stage_type == "GroupConvolution" ||
         stage_type == "GroupConv2D";
}

bool apple_requires_scalar_fp32_convolution_path(
    const std::shared_ptr<const ov::Node> &node) {
  return node && node->get_output_size() > 0 &&
         node->get_output_element_type(0) == ov::element::f32 &&
         ov::fp16_compression_is_disabled(node);
}

MlirStageBackendSourcePlan
to_backend_source_plan(const GfxMslGeneratedKernelSourcePlan &plan) {
  if (!plan.valid()) {
    return {};
  }
  MlirStageBackendSourcePlan result;
  result.source = plan.source;
  result.runtime_binding = plan.binding.runtime_binding;
  result.has_runtime_binding = plan.binding.valid;
  return result;
}

MlirStageBackendSourcePlan
to_backend_source_plan(const GfxMpsrtKernelSourcePlan &plan) {
  if (!plan.valid()) {
    return {};
  }
  MlirStageBackendSourcePlan result;
  result.source = plan.source;
  result.runtime_binding = plan.runtime_binding;
  result.has_runtime_binding = plan.has_runtime_binding;
  result.requires_backend_model = plan.requires_mpsrt_model;
  return result;
}

MlirStageBackendRuntimeParamsPlan
to_backend_runtime_params_plan(const GfxSdpaMslRuntimeParamsPlan &plan) {
  if (!plan.valid()) {
    return {};
  }
  MlirStageBackendRuntimeParamsPlan result;
  result.valid = true;
  result.params = plan.params;
  result.runtime_binding = plan.binding.runtime_binding;
  return result;
}

class AppleMlirStageBackendHooks final : public MlirStageBackendHooks {
public:
  GfxKernelBackendDomain custom_kernel_backend_domain() const override {
    return GfxKernelBackendDomain::AppleMsl;
  }

  bool should_pack_matmul_const_input_as_f16(
      const std::shared_ptr<const ov::Node> &node, size_t input_idx,
      const ov::Tensor &tensor) const override {
    auto matmul = ov::as_type_ptr<const ov::op::v0::MatMul>(node);
    return matmul && input_idx == 1 &&
           (!matmul->get_input_partial_shape(0).is_static() ||
            !matmul->get_output_partial_shape(0).is_static()) &&
           tensor.get_element_type() == ov::element::f32 &&
           matmul->get_output_element_type(0) == ov::element::f32;
  }

  bool should_pack_conv2d_const_weights_oc4(
      const std::shared_ptr<const ov::Node> &node, size_t input_idx,
      const ov::Tensor &tensor) const override {
    auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(node);
    if (!conv || input_idx != 1 || tensor.get_shape().size() != 4) {
      return false;
    }
    if (apple_requires_scalar_fp32_convolution_path(node)) {
      return false;
    }
    const auto et = tensor.get_element_type();
    if (et != ov::element::f16 && et != ov::element::f32) {
      return false;
    }
    if (!conv->get_input_partial_shape(0).is_static() ||
        !conv->get_input_partial_shape(1).is_static() ||
        !conv->get_output_partial_shape(0).is_static()) {
      return false;
    }
    const auto in_shape = conv->get_input_shape(0);
    const auto w_shape = conv->get_input_shape(1);
    const auto out_shape = conv->get_output_shape(0);
    if (in_shape.size() != 4 || w_shape.size() != 4 ||
        out_shape.size() != 4) {
      return false;
    }
    Conv2DCodegenDesc desc{};
    desc.input_type = conv->get_input_element_type(0);
    desc.weight_type = conv->get_input_element_type(1);
    desc.output_type = conv->get_output_element_type(0);
    desc.C_out = static_cast<uint32_t>(w_shape[0]);
    desc.kH = static_cast<uint32_t>(w_shape[2]);
    desc.kW = static_cast<uint32_t>(w_shape[3]);
    const uint32_t cin_pg = static_cast<uint32_t>(w_shape[1]);
    const uint32_t c_in = static_cast<uint32_t>(in_shape[1]);
    desc.groups = (cin_pg != 0 && c_in % cin_pg == 0) ? (c_in / cin_pg) : 1;
    return gfx_conv2d_output_channel_block(desc) >= 4;
  }

  void attach_const_tensor_sources(
      KernelSource &source,
      const std::shared_ptr<const ov::Node> &node) const override {
    gfx_attach_mpsrt_const_tensor_sources(source, node);
  }

  bool apply_stage_optimization(
      mlir::ModuleOp module, const GfxStageOptimizationPlan &plan,
      const std::shared_ptr<const ov::Node> &node,
      std::string_view stage_type, bool has_bias,
      const BiasParams *bias_params, bool has_activation,
      ActivationKind activation) const override {
    if (!module) {
      return false;
    }

    const bool conv_like = is_apple_conv_like_stage(stage_type);
    const bool conv_mpsrt_annotated =
        conv_like &&
        annotate_module_with_conv_mpsrt_plan(
            module, plan, node, stage_type, has_bias, bias_params,
            has_activation, activation) != GfxConvMpsrtLoweringKind::None;
    const bool deferred_mps_conv_materialization =
        conv_like && !conv_mpsrt_annotated &&
        plan.placement.domain == GfxStageBackendDomain::AppleMps;
    if (!conv_mpsrt_annotated && !deferred_mps_conv_materialization) {
      const bool materialize_typed_program =
          plan.placement.domain == GfxStageBackendDomain::AppleMps ||
          plan.placement.domain == GfxStageBackendDomain::AppleMsl;
      (void)run_gfx_apple_stage_pipeline(module, plan, std::string(stage_type),
                                         {}, materialize_typed_program);
      return materialize_typed_program;
    }
    return true;
  }

  uint32_t static_matmul_reduction_threads(
      const std::shared_ptr<const ov::Node> &node, bool has_activation,
      ActivationKind activation, float activation_alpha) const override {
    auto desc = make_static_matmul_codegen_desc_for_node(node);
    if (!desc) {
      return 0;
    }
    desc->has_activation = has_activation;
    desc->activation = activation;
    desc->alpha = activation_alpha;
    return gfx_matmul_parallel_reduction_threads(*desc);
  }

  MlirStageBackendCompressedMatMulPlan make_compressed_matmul_plan(
      const std::shared_ptr<const ov::Node> &node,
      const GfxParallelismCaps &caps) const override {
    auto info = detect_compressed_matmul_weights(node);
    if (!info) {
      return {};
    }

    MlirStageBackendCompressedMatMulPlan result;
    result.reduction_threads =
        compressed_matmul_parallel_reduction_threads(*info, caps);
    result.output_block =
        compressed_matmul_output_block(*info, caps, result.reduction_threads);
    result.output_columns = static_cast<uint32_t>(info->n);
    result.packed_weights =
        pack_compressed_matmul_weights_for_output_block(*info,
                                                       result.output_block);
    result.packed_scales = pack_compressed_matmul_scales(*info);
    result.packed_scale_type = info->scale->get_element_type();
    result.packed_scale_shape = ov::Shape{info->n, info->groups, 1};
    result.packed_weight_suffix =
        std::string("compressed_matmul/packed_weights/") +
        info->weights->get_element_type().get_type_name() + "/block" +
        std::to_string(result.output_block);
    result.source_plan = to_backend_source_plan(
        make_compressed_matmul_msl_kernel_source_plan(
            *info, result.reduction_threads, result.output_block));
    result.valid = result.source_plan.valid();
    return result;
  }

  MlirStageBackendSourcePlan make_shapeof_source_plan(
      const std::shared_ptr<const ov::Node> &node,
      mlir::ModuleOp module) const override {
    return to_backend_source_plan(make_shapeof_msl_kernel_source_plan(node,
                                                                      module));
  }

  MlirStageBackendSourcePlan make_concat_source_plan(
      const std::shared_ptr<const ov::Node> &node,
      mlir::ModuleOp module) const override {
    return to_backend_source_plan(make_concat_msl_kernel_source_plan(node,
                                                                     module));
  }

  MlirStageBackendSourcePlan
  make_causal_sdpa_source_plan(ov::element::Type element_type) const override {
    return to_backend_source_plan(
        make_causal_sdpa_msl_kernel_source_plan(element_type));
  }

  MlirStageBackendSourcePlan make_sdpa_source_plan(
      mlir::MLIRContext &ctx, const std::shared_ptr<const ov::Node> &node,
      const GpuBufferManager *buffer_manager,
      const GfxStageRuntimeTraits &runtime_traits) const override {
    KernelSource sdpa_mps_source;
    sdpa_mps_source.module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    auto mps_source_plan = configure_apple_mps_vendor_kernel_source_plan_for_node(
        sdpa_mps_source, node, buffer_manager, "ScaledDotProductAttention",
        /*has_bias=*/false,
        /*has_activation=*/false,
        /*has_batchnorm=*/false, ActivationKind::Identity, nullptr,
        runtime_traits);
    if (mps_source_plan.valid()) {
      return to_backend_source_plan(mps_source_plan);
    }

    const auto et = node ? node->get_output_element_type(0)
                         : ov::element::dynamic;
    const bool has_mask = node && node->get_input_size() >= 4;
    return to_backend_source_plan(make_sdpa_msl_kernel_source_plan(et,
                                                                   has_mask));
  }

  MlirStageBackendSourcePlan make_range_source_plan(
      const std::shared_ptr<const ov::Node> &node,
      mlir::ModuleOp module) const override {
    return to_backend_source_plan(make_range_msl_kernel_source_plan(node,
                                                                    module));
  }

  MlirStageBackendSourcePlan make_tile_source_plan(
      const std::shared_ptr<const ov::Node> &node,
      mlir::ModuleOp module) const override {
    return to_backend_source_plan(make_tile_msl_kernel_source_plan(node,
                                                                   module));
  }

  MlirStageBackendSourcePlan make_activation_source_plan(
      const std::shared_ptr<const ov::Node> &node,
      mlir::ModuleOp module) const override {
    return to_backend_source_plan(make_activation_msl_kernel_source_plan(node,
                                                                         module));
  }

  MlirStageBackendSourcePlan make_runtime_matmul_source_plan(
      mlir::MLIRContext &ctx, const GpuBufferManager *buffer_manager,
      const std::shared_ptr<const ov::Node> &node,
      const MatMulCodegenDesc &desc, const ov::Shape &shape_a,
      const ov::Shape &shape_b, std::string_view stage_name) const override {
    return to_backend_source_plan(make_apple_metal_runtime_matmul_kernel_source_plan(
        ctx, buffer_manager, node, desc, shape_a, shape_b, stage_name));
  }

  MlirStageBackendSourcePlan make_static_slice_source_plan(
      const std::shared_ptr<const ov::Node> &node,
      const ov::element::Type &storage_type,
      mlir::ModuleOp module) const override {
    return to_backend_source_plan(
        make_direct_static_slice_msl_kernel_source_plan(node, storage_type,
                                                        module));
  }

  MlirStageBackendSourcePlan make_direct_split_source_plan(
      std::string_view stage_type, const ov::element::Type &element_type,
      const ov::Shape &input_shape, const std::vector<size_t> &split_sizes,
      uint32_t axis_len, uint32_t inner_stride,
      mlir::ModuleOp module) const override {
    return to_backend_source_plan(make_direct_split_msl_kernel_source_plan(
        stage_type, element_type, input_shape, split_sizes, axis_len,
        inner_stride, module));
  }

  MlirStageBackendRuntimeParamsPlan make_causal_sdpa_runtime_params_plan(
      const ov::Shape &q_shape, const ov::Shape &k_shape,
      const ov::Shape &v_shape, const ov::Shape &mask_shape, float scale,
      bool k_gqa, size_t k_heads, bool v_gqa,
      size_t v_heads) const override {
    return to_backend_runtime_params_plan(
        make_causal_sdpa_msl_runtime_params_plan(
            q_shape, k_shape, v_shape, mask_shape, scale, k_gqa, k_heads,
            v_gqa, v_heads));
  }

  MlirStageBackendRuntimeParamsPlan make_sdpa_runtime_params_plan(
      const ov::Shape &q_shape, const ov::Shape &k_shape,
      const ov::Shape &v_shape, const ov::Shape &mask_shape, bool has_mask,
      float scale, bool k_gqa, size_t k_heads, bool v_gqa,
      size_t v_heads) const override {
    return to_backend_runtime_params_plan(make_sdpa_msl_runtime_params_plan(
        q_shape, k_shape, v_shape, mask_shape, has_mask, scale, k_gqa, k_heads,
        v_gqa, v_heads));
  }

  MlirStageBackendReductionPlan make_reduction_plan(
      const std::shared_ptr<const ov::Node> &node) const override {
    const auto kind = reduction_kind_from_node(node);
    if (!kind) {
      return {};
    }
    MlirStageBackendReductionPlan plan;
    plan.valid = true;
    plan.op_code = reduction_kernel_op_code(*kind);
    plan.entry_point = std::string(reduction_msl_kernel_entry_point(*kind));
    return plan;
  }
};

} // namespace

const MlirStageBackendHooks &apple_mlir_stage_backend_hooks() {
  static const AppleMlirStageBackendHooks hooks;
  return hooks;
}

void ensure_apple_mlir_stage_backend_hooks_registered() {
  static const bool registered = register_mlir_stage_backend_hooks(
      GpuBackend::Metal, apple_mlir_stage_backend_hooks());
  (void)registered;
}

namespace {
const bool kAppleMlirStageBackendHooksRegistered =
    (ensure_apple_mlir_stage_backend_hooks_registered(), true);
} // namespace

} // namespace gfx_plugin
} // namespace ov
