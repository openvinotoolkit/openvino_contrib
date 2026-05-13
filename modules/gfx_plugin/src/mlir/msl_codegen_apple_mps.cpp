// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen_apple_mps.hpp"

#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "kernel_ir/gfx_kernel_manifest.hpp"
#include "mlir/codegen_common.hpp"
#include "mlir/gfx_apple_stage_pipeline.hpp"
#include "mlir/gfx_apple_vendor_descriptors.hpp"
#include "mlir/gfx_backend_custom_kernel_adapter.hpp"
#include "mlir/gfx_mlir_kernel_builder.hpp"
#include "mlir/gfx_mpsrt_const_tensor_sources.hpp"
#include "mlir/gfx_mpsrt_conv_metadata.hpp"
#include "mlir/gfx_mpsrt_metadata.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/topk.hpp"
#include "runtime/gfx_stage_policy.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

bool has_apple_msl_custom_kernel_manifest(mlir::ModuleOp module) {
  GfxKernelStageManifest manifest{};
  return module &&
         detail::gfx_mpsrt_read_stage_manifest_attrs(module, manifest) &&
         manifest.valid &&
         manifest.backend_domain == GfxKernelBackendDomain::AppleMsl &&
         manifest.execution_kind == GfxKernelExecutionKind::CustomKernel;
}

bool conv_bias_is_channel_vector(const BiasParams *bias_params,
                                 const std::shared_ptr<const ov::Node> &node) {
  if (!bias_params || bias_params->empty() || !node ||
      !node->get_output_partial_shape(0).is_static()) {
    return false;
  }
  const auto output_shape = node->get_output_shape(0);
  return output_shape.size() == 4 && output_shape[1] != 0 &&
         bias_params->values.size() == output_shape[1];
}

void attach_conv_bias_const_tensor(KernelSource &source,
                                   const BiasParams *bias_params) {
  if (!bias_params || bias_params->empty()) {
    return;
  }
  GfxMpsrtProgram program{};
  if (!read_module_mpsrt_program(source.module, program) || !program.valid ||
      program.inputs.size() < 3 ||
      !gfx_mpsrt_program_input_is_const(program, 2) ||
      gfx_mpsrt_const_payload_already_attached(source, 2)) {
    return;
  }
  MpsrtConstTensorSource payload{};
  payload.value = 2;
  const auto dtype = program.inputs[2].dtype;
  if (dtype == GfxMpsrtDType::F16) {
    std::vector<ov::float16> values;
    values.reserve(bias_params->values.size());
    for (const auto value : bias_params->values) {
      values.emplace_back(value);
    }
    payload.bytes.resize(values.size() * sizeof(ov::float16));
    std::memcpy(payload.bytes.data(), values.data(), payload.bytes.size());
  } else if (dtype == GfxMpsrtDType::F32) {
    payload.bytes.resize(bias_params->values.size() * sizeof(float));
    std::memcpy(payload.bytes.data(), bias_params->values.data(),
                payload.bytes.size());
  } else {
    return;
  }
  source.mpsrt_const_tensors.push_back(std::move(payload));
}

KernelRuntimeBindingState make_conv_mpsrt_runtime_binding(bool has_bias) {
  const size_t input_arg_count = has_bias ? 3u : 2u;
  std::vector<int32_t> operand_kinds(has_bias ? 4u : 3u, 1);
  std::vector<int32_t> operand_arg_indices;
  operand_arg_indices.reserve(operand_kinds.size());
  operand_arg_indices.push_back(0);
  operand_arg_indices.push_back(1);
  if (has_bias) {
    operand_arg_indices.push_back(2);
  }
  operand_arg_indices.push_back(static_cast<int32_t>(input_arg_count));
  return make_kernel_runtime_binding_state({0}, input_arg_count,
                                           std::move(operand_kinds),
                                           std::move(operand_arg_indices));
}

std::vector<int64_t> shape_to_i64_vector(const ov::Shape &shape) {
  std::vector<int64_t> dims;
  dims.reserve(shape.size());
  for (const auto dim : shape) {
    dims.push_back(static_cast<int64_t>(dim));
  }
  return dims;
}

GfxMpsrtStageDesc make_conv_texture_swish_epilogue_stage_desc() {
  const auto binding = make_backend_custom_kernel_roles_binding_plan(
      "ConvTextureSwishEpilogue", "gfx_mpsrt_conv_texture_swish_epilogue",
      {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput});

  GfxMpsrtStageDesc stage{};
  stage.kind = GfxMpsrtStageKind::MSLDispatch;
  stage.domain = GfxStageBackendDomain::AppleMsl;
  stage.input_storage = GfxMpsrtStorage::Image;
  stage.output_storage = GfxMpsrtStorage::Buffer;
  stage.layout = GfxMpsrtLayout::Linear;
  stage.kernel_name = "gfx_mpsrt_conv_texture_swish_epilogue";
  if (binding.valid) {
    stage.stage_manifest = binding.stage_manifest;
  }
  return stage;
}

std::string generate_conv_texture_swish_epilogue_msl(
    const std::shared_ptr<const ov::Node> &node) {
  const auto output_shape = node->get_output_shape(0);
  const auto output_type = node->get_output_element_type(0);
  const std::string scalar = msl_type_from_element(output_type);

  std::ostringstream ss;
  ss << "#include <metal_stdlib>\n";
  ss << "using namespace metal;\n";
  ss << "constant uint N = " << output_shape[0] << ";\n";
  ss << "constant uint C = " << output_shape[1] << ";\n";
  ss << "constant uint H = " << output_shape[2] << ";\n";
  ss << "constant uint W = " << output_shape[3] << ";\n";
  ss << "inline float gfx_mpsrt_swish(float x) {\n";
  ss << "  return x / (1.0f + precise::exp(-x));\n";
  ss << "}\n";
  ss << "kernel void gfx_mpsrt_conv_texture_swish_epilogue(\n";
  ss << "  texture2d_array<" << scalar
     << ", access::read> input [[texture(0)]],\n";
  ss << "  device " << scalar << "* output [[buffer(0)]],\n";
  ss << "  uint gid [[thread_position_in_grid]]) {\n";
  ss << "  const uint total = N * C * H * W;\n";
  ss << "  if (gid >= total) return;\n";
  ss << "  const uint x = gid % W;\n";
  ss << "  const uint yh = gid / W;\n";
  ss << "  const uint y = yh % H;\n";
  ss << "  const uint ch = yh / H;\n";
  ss << "  const uint c = ch % C;\n";
  ss << "  const uint n = ch / C;\n";
  ss << "  const uint slices = (C + 3u) / 4u;\n";
  ss << "  const uint plane = n * slices + c / 4u;\n";
  ss << "  const uint lane = c & 3u;\n";
  ss << "  const auto v = input.read(uint2(x, y), plane);\n";
  ss << "  float value = (lane == 0u) ? static_cast<float>(v.x) :\n";
  ss << "                (lane == 1u) ? static_cast<float>(v.y) :\n";
  ss << "                (lane == 2u) ? static_cast<float>(v.z) : "
        "static_cast<float>(v.w);\n";
  ss << "  value = gfx_mpsrt_swish(value);\n";
  ss << "  output[gid] = static_cast<" << scalar << ">(value);\n";
  ss << "}\n";
  return ss.str();
}

GfxMpsrtKernelSourcePlan try_make_conv_texture_swish_mpsrt_source_plan(
    KernelSource source, const std::shared_ptr<const ov::Node> &node,
    const GfxStageOptimizationPlan &plan, std::string_view stage_type,
    bool has_bias, const BiasParams *bias_params) {
  if (!source.module || !node ||
      !node->get_output_partial_shape(0).is_static() ||
      node->get_output_shape(0).size() != 4 ||
      (node->get_output_element_type(0) != ov::element::f16 &&
       node->get_output_element_type(0) != ov::element::f32)) {
    return {};
  }

  GfxAppleMpsVendorPrimitiveContract contract{};
  if (!gfx_apple_make_mps_conv2d_contract(node, has_bias, bias_params,
                                          /*has_activation=*/false,
                                          ActivationKind::Identity, contract) ||
      contract.input_descs.size() < 2 || contract.output_descs.size() != 1) {
    return {};
  }

  const bool group_conv = ov::is_type<const ov::op::v1::GroupConvolution>(node);
  const char *canonical_stage_type =
      group_conv ? "GroupConvolution" : "Convolution";
  auto conv_stage = gfx_mpsrt_make_stage_desc(plan, canonical_stage_type);
  conv_stage.conv2d_desc = contract.descriptor.conv2d;
  if (conv_stage.kind != GfxMpsrtStageKind::MPSConv2D &&
      conv_stage.kind != GfxMpsrtStageKind::MPSGroupConv2D) {
    return {};
  }

  GfxAppleMpsrtProgramPlanBuilder builder(
      std::string("mps_conv2d_plus_msl_texture_swish_model|") +
      std::string(stage_type));
  std::vector<GfxMpsrtValue> conv_inputs;
  conv_inputs.reserve(contract.input_descs.size());
  conv_inputs.push_back(builder.add_external_input(
      contract.input_descs[0], GfxMpsrtExternalBufferRole::TensorInput));
  conv_inputs.push_back(builder.add_external_input(
      contract.input_descs[1], GfxMpsrtExternalBufferRole::ConstBuffer));
  if (has_bias) {
    if (contract.input_descs.size() != 3) {
      return {};
    }
    conv_inputs.push_back(builder.add_external_input(
        contract.input_descs[2], GfxMpsrtExternalBufferRole::ConstBuffer));
  }

  const auto conv_output = builder.add_single_output_stage(
      conv_stage, conv_inputs, contract.output_descs.front());
  const auto final_output_desc = gfx_mpsrt_make_tensor_desc(
      shape_to_i64_vector(node->get_output_shape(0)),
      node->get_output_element_type(0), GfxStageStorageKind::Buffer,
      GfxMpsrtTensorFlagExternalIo);
  const auto output_value = builder.add_single_output_stage(
      make_conv_texture_swish_epilogue_stage_desc(), {conv_output},
      final_output_desc);
  builder.add_external_output(output_value);

  const auto program_plan = builder.finalize();
  const auto materialized =
      materialize_apple_mpsrt_program_plan(source.module, program_plan);
  if (!materialized.valid || !materialized.typed_program_materialized) {
    return {};
  }
  auto source_plan = make_mpsrt_kernel_source_plan_from_msl_source(
      source.module, generate_conv_texture_swish_epilogue_msl(node));
  if (!source_plan.valid()) {
    return {};
  }
  gfx_attach_mpsrt_const_tensors(source_plan.source, node);
  if (has_bias) {
    attach_conv_bias_const_tensor(source_plan.source, bias_params);
  }
  source_plan.runtime_binding = make_conv_mpsrt_runtime_binding(has_bias);
  source_plan.has_runtime_binding = true;
  return source_plan;
}

GfxMpsrtKernelSourcePlan
try_configure_apple_mps_vendor_kernel_source_plan_for_node(
    KernelSource source, const std::shared_ptr<const ov::Node> &node,
    const GpuBufferManager *buffer_manager, std::string_view stage_type,
    bool has_bias, bool has_activation, bool has_batchnorm,
    ActivationKind activation, const BiasParams *bias_params,
    const GfxStageRuntimeTraits &traits) {
  if (!source.module || !node) {
    return {};
  }
  if (has_apple_msl_custom_kernel_manifest(source.module)) {
    return {};
  }

  const bool conv_base_candidate =
      (ov::is_type<const ov::op::v1::Convolution>(node) ||
       ov::is_type<const ov::op::v1::GroupConvolution>(node)) &&
      (!has_bias || conv_bias_is_channel_vector(bias_params, node)) &&
      !has_batchnorm;
  if (conv_base_candidate && has_activation &&
      activation == ActivationKind::Swish) {
    const bool group_conv =
        ov::is_type<const ov::op::v1::GroupConvolution>(node);
    const char *canonical_stage_type =
        group_conv ? "GroupConvolution" : "Convolution";
    const auto plan = select_stage_optimization_plan(
        buffer_manager, GpuBackend::Metal, canonical_stage_type, node,
        node->get_output_element_type(0), has_bias,
        /*has_activation=*/false,
        /*has_batchnorm=*/false, traits);
    if (plan.placement.domain == GfxStageBackendDomain::AppleMps &&
        plan.placement.storage == GfxStageStorageKind::Image &&
        plan.placement.uses_vendor_primitive) {
      auto source_plan = try_make_conv_texture_swish_mpsrt_source_plan(
          std::move(source), node, plan, canonical_stage_type, has_bias,
          bias_params);
      if (source_plan.valid()) {
        return source_plan;
      }
    }
    return {};
  }

  const bool conv_candidate =
      conv_base_candidate &&
      (!has_activation || gfx_mpsrt_conv_supports_fused_activation(activation));
  if (conv_candidate) {
    const bool group_conv =
        ov::is_type<const ov::op::v1::GroupConvolution>(node);
    const char *canonical_stage_type =
        group_conv ? "GroupConvolution" : "Convolution";
    const char *fallback_stage_type =
        group_conv ? "GroupConv2D" : "Convolution";
    const auto plan = select_stage_optimization_plan(
        buffer_manager, GpuBackend::Metal, canonical_stage_type, node,
        node->get_output_element_type(0), has_bias, has_activation,
        /*has_batchnorm=*/false, traits);
    const auto lowering = annotate_module_with_conv_mpsrt_plan(
        source.module, plan, node, fallback_stage_type, has_bias, bias_params,
        has_activation, activation);
    if (lowering == GfxConvMpsrtLoweringKind::MpsConv2D ||
        lowering == GfxConvMpsrtLoweringKind::MpsGroupConv2D) {
      auto source_plan =
          make_mpsrt_kernel_source_plan_from_module(source.module);
      if (source_plan.valid()) {
        gfx_attach_mpsrt_const_tensors(source_plan.source, node);
        if (has_bias) {
          attach_conv_bias_const_tensor(source_plan.source, bias_params);
        }
        source_plan.runtime_binding = make_conv_mpsrt_runtime_binding(has_bias);
        source_plan.has_runtime_binding = true;
        return source_plan;
      }
    }
  }

  if (stage_type == "MaxPool" || stage_type == "AvgPool") {
    GfxMpsrtPool2DAbiDesc pool_desc{};
    if (gfx_apple_make_mps_pool2d_desc(node, pool_desc)) {
      const auto plan = select_stage_optimization_plan(
          buffer_manager, GpuBackend::Metal, std::string(stage_type), node,
          node->get_output_element_type(0),
          /*has_bias=*/false,
          /*has_activation=*/false,
          /*has_batchnorm=*/false, traits);
      if (plan.placement.domain == GfxStageBackendDomain::AppleMps &&
          plan.placement.storage == GfxStageStorageKind::Image) {
        GfxAppleMpsVendorPrimitiveContract contract{};
        if (!gfx_apple_make_mps_pool2d_contract(node, pool_desc, contract)) {
          return {};
        }
        auto source_module = source.module;
        const auto materialized = materialize_apple_mps_vendor_contract_program(
            source_module, plan, std::string(stage_type), contract);
        if (materialized.valid) {
          auto source_plan =
              make_mpsrt_kernel_source_plan_from_module(source_module);
          if (source_plan.valid()) {
            return source_plan;
          }
        }
      }
    }
  }

  if (stage_type == "Interpolate") {
    GfxMpsrtResize2DAbiDesc resize_desc{};
    if (gfx_apple_make_mps_resize2d_desc(node, resize_desc)) {
      const auto plan = select_stage_optimization_plan(
          buffer_manager, GpuBackend::Metal, "Interpolate", node,
          node->get_output_element_type(0),
          /*has_bias=*/false,
          /*has_activation=*/false,
          /*has_batchnorm=*/false, traits);
      if (plan.placement.domain == GfxStageBackendDomain::AppleMps &&
          plan.placement.storage == GfxStageStorageKind::Image) {
        GfxAppleMpsVendorPrimitiveContract contract{};
        if (!gfx_apple_make_mps_resize2d_contract(node, resize_desc,
                                                  contract)) {
          return {};
        }
        auto source_module = source.module;
        const auto materialized = materialize_apple_mps_vendor_contract_program(
            source_module, plan, "Interpolate", contract);
        if (materialized.valid) {
          auto source_plan =
              make_mpsrt_kernel_source_plan_from_module(source_module);
          if (source_plan.valid()) {
            return source_plan;
          }
        }
      }
    }
  }

  if (stage_type == "Softmax") {
    GfxMpsrtSoftmaxAbiDesc softmax_desc{};
    if (gfx_apple_make_mps_softmax_desc(node, softmax_desc)) {
      const auto plan = select_stage_optimization_plan(
          buffer_manager, GpuBackend::Metal, "Softmax", node,
          node->get_output_element_type(0),
          /*has_bias=*/false,
          /*has_activation=*/false,
          /*has_batchnorm=*/false, traits);
      if (plan.placement.domain == GfxStageBackendDomain::AppleMps &&
          plan.placement.storage == GfxStageStorageKind::Matrix) {
        GfxAppleMpsVendorPrimitiveContract contract{};
        if (!gfx_apple_make_mps_softmax_contract(node, softmax_desc,
                                                 contract)) {
          return {};
        }
        auto source_module = source.module;
        const auto materialized = materialize_apple_mps_vendor_contract_program(
            source_module, plan, "Softmax", contract);
        if (materialized.valid) {
          auto source_plan =
              make_mpsrt_kernel_source_plan_from_module(source_module);
          if (source_plan.valid()) {
            return source_plan;
          }
        }
      }
    }
  }

  if (stage_type == "TopK") {
    GfxMpsrtTopKAbiDesc topk_desc{};
    if (gfx_apple_make_mps_topk_desc(node, topk_desc)) {
      const auto plan = select_stage_optimization_plan(
          buffer_manager, GpuBackend::Metal, "TopK", node,
          node->get_output_element_type(0),
          /*has_bias=*/false,
          /*has_activation=*/false,
          /*has_batchnorm=*/false, traits);
      if (plan.placement.domain == GfxStageBackendDomain::AppleMps &&
          plan.placement.storage == GfxStageStorageKind::Matrix) {
        GfxAppleMpsVendorPrimitiveContract contract{};
        if (!gfx_apple_make_mps_topk_contract(node, topk_desc, contract)) {
          return {};
        }
        auto source_module = source.module;
        const auto materialized = materialize_apple_mps_vendor_contract_program(
            source_module, plan, "TopK", contract);
        if (materialized.valid) {
          auto source_plan =
              make_mpsrt_kernel_source_plan_from_module(source_module);
          if (source_plan.valid()) {
            return source_plan;
          }
        }
      }
    }
  }

  return {};
}

GfxMpsrtKernelSourcePlan
try_configure_clean_apple_mps_vendor_kernel_source_plan_for_node(
    const std::shared_ptr<const ov::Node> &node,
    const GpuBufferManager *buffer_manager, std::string_view stage_type,
    bool has_bias, bool has_activation, bool has_batchnorm,
    ActivationKind activation, const BiasParams *bias_params,
    const GfxStageRuntimeTraits &traits) {
  if (!node) {
    return {};
  }

  KernelSource clean_source;
  clean_source.module = build_mlir_for_node(node, gfx_mlir_context());
  if (!clean_source.module) {
    return {};
  }
  return try_configure_apple_mps_vendor_kernel_source_plan_for_node(
      clean_source, node, buffer_manager, stage_type, has_bias, has_activation,
      has_batchnorm, activation, bias_params, traits);
}

} // namespace

GfxMpsrtKernelSourcePlan configure_apple_mps_vendor_kernel_source_plan_for_node(
    KernelSource source, const std::shared_ptr<const ov::Node> &node,
    const GpuBufferManager *buffer_manager, std::string_view stage_type,
    bool has_bias, bool has_activation, bool has_batchnorm,
    ActivationKind activation, const BiasParams *bias_params,
    const GfxStageRuntimeTraits &traits) {
  auto source_plan = try_configure_apple_mps_vendor_kernel_source_plan_for_node(
      source, node, buffer_manager, stage_type, has_bias, has_activation,
      has_batchnorm, activation, bias_params, traits);
  if (source_plan.valid() || !source.module ||
      !has_apple_msl_custom_kernel_manifest(source.module)) {
    return source_plan;
  }

  return try_configure_clean_apple_mps_vendor_kernel_source_plan_for_node(
      node, buffer_manager, stage_type, has_bias, has_activation, has_batchnorm,
      activation, bias_params, traits);
}

} // namespace gfx_plugin
} // namespace ov
