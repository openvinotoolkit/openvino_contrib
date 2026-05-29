// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen_apple_msl_softmax.hpp"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "kernel_ir/metal_kernels/softmax_kernels.hpp"
#include "mlir/codegen_common.hpp"
#include "mlir/gfx_backend_custom_kernel_adapter.hpp"
#include "mlir/mlir_builder.hpp"
#include "mlir/mlir_support.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/softmax.hpp"
#include "runtime/gfx_shape_utils.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

std::optional<int64_t>
softmax_axis_from_node(const std::shared_ptr<const ov::Node> &node) {
  if (!node) {
    return std::nullopt;
  }

  if (auto sm1 = ov::as_type_ptr<const ov::op::v1::Softmax>(node)) {
    return sm1->get_axis();
  }
  if (auto sm8 = ov::as_type_ptr<const ov::op::v8::Softmax>(node)) {
    return sm8->get_axis();
  }
  if (auto ls = ov::as_type_ptr<const ov::op::v5::LogSoftmax>(node)) {
    return ls->get_axis();
  }
  return std::nullopt;
}

bool softmax_is_log(const std::shared_ptr<const ov::Node> &node) {
  return static_cast<bool>(ov::as_type_ptr<const ov::op::v5::LogSoftmax>(node));
}

bool softmax_msl_type_supported(const std::shared_ptr<const ov::Node> &node) {
  if (!node || node->get_input_size() != 1 || node->get_output_size() != 1 ||
      !node->get_input_partial_shape(0).is_static() ||
      !node->get_output_partial_shape(0).is_static() ||
      node->get_input_shape(0) != node->get_output_shape(0) ||
      node->get_input_element_type(0) != node->get_output_element_type(0)) {
    return false;
  }
  const auto type = node->get_output_element_type(0);
  return type == ov::element::f32 || type == ov::element::f16;
}

const GfxKernelSource &
softmax_msl_source(const SoftmaxMslKernelDescriptor &descriptor) {
  if (descriptor.kernel_unit_id == "metal/generated/softmax_f32") {
    return metal_generated_softmax_f32_kernel_source();
  }
  if (descriptor.kernel_unit_id == "metal/generated/softmax_f16") {
    return metal_generated_softmax_f16_kernel_source();
  }
  if (descriptor.kernel_unit_id == "metal/generated/logsoftmax_f32") {
    return metal_generated_logsoftmax_f32_kernel_source();
  }
  return metal_generated_logsoftmax_f16_kernel_source();
}

std::optional<GfxKernelRuntimeBindingPlan>
existing_softmax_runtime_params_binding(const KernelSource &source) {
  GfxKernelStageManifest manifest{};
  if (!read_backend_custom_kernel_stage_manifest_from_module(
          source.module, GfxKernelBackendDomain::AppleMsl, manifest)) {
    return std::nullopt;
  }
  auto binding =
      make_backend_custom_kernel_binding_plan_from_stage_manifest(manifest);
  if (!binding.valid ||
      manifest.stage_family != GfxKernelStageFamily::Softmax) {
    return std::nullopt;
  }
  const auto roles = materialize_gfx_kernel_external_buffer_roles(
      manifest.custom_kernel.external_buffer_abi);
  for (const auto role : roles) {
    if (role == GfxKernelBufferRole::RuntimeParams) {
      return binding;
    }
  }
  return std::nullopt;
}

std::optional<KernelSource> make_runtime_params_softmax_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node,
    const std::optional<ov::Shape> &runtime_input_shape,
    const GfxKernelRuntimeBindingPlan &binding) {
  if (!node || !source.module || !softmax_axis_from_node(node)) {
    return std::nullopt;
  }
  ov::Shape input_shape;
  if (runtime_input_shape && !runtime_input_shape->empty()) {
    input_shape = *runtime_input_shape;
  } else if (node->get_input_partial_shape(0).is_static()) {
    input_shape = node->get_input_shape(0);
  }
  if (input_shape.empty()) {
    return std::nullopt;
  }

  const auto dims = compute_softmax_dims(
      input_shape, *softmax_axis_from_node(node), "GFX Metal Softmax");
  SoftmaxCodegenDesc desc{};
  desc.element_type = node->get_output_element_type(0);
  desc.rows = static_cast<int64_t>(dims.rows);
  desc.cols = static_cast<int64_t>(dims.axis_len);
  desc.inner = static_cast<int64_t>(dims.inner);
  desc.log_softmax = softmax_is_log(node);

  source.entry_point = binding.stage_manifest.custom_kernel.entry_point;
  source.msl_source.clear();
  source.msl_generator = [desc](mlir::ModuleOp module) mutable {
    return generate_msl_from_mlir(module, desc);
  };
  if (!configure_backend_custom_kernel_source_from_binding_plan(source,
                                                                binding)) {
    return std::nullopt;
  }
  return source;
}

mlir::ModuleOp
make_softmax_codegen_module(const std::shared_ptr<const ov::Node> &node,
                            mlir::ModuleOp module,
                            const ov::Shape &input_shape) {
  if (module) {
    return module;
  }
  auto &ctx = gfx_mlir_context();
  return softmax_is_log(node)
             ? build_mlir_logsoftmax_from_node(node, ctx, input_shape)
             : build_mlir_softmax_from_node(node, ctx, input_shape);
}

} // namespace

std::optional<SoftmaxMslKernelDescriptor>
softmax_msl_kernel_descriptor(const std::shared_ptr<const ov::Node> &node) {
  if (!node || !softmax_axis_from_node(node)) {
    return std::nullopt;
  }

  const bool log_softmax = softmax_is_log(node);
  const auto type = node->get_output_element_type(0);
  if (log_softmax) {
    if (type == ov::element::f32) {
      return SoftmaxMslKernelDescriptor{"metal/generated/logsoftmax_f32",
                                        "gfx_metal_generated_logsoftmax_f32",
                                        true};
    }
    if (type == ov::element::f16) {
      return SoftmaxMslKernelDescriptor{"metal/generated/logsoftmax_f16",
                                        "gfx_metal_generated_logsoftmax_f16",
                                        true};
    }
    return std::nullopt;
  }
  if (type == ov::element::f32) {
    return SoftmaxMslKernelDescriptor{"metal/generated/softmax_f32",
                                      "gfx_metal_generated_softmax_f32", false};
  }
  if (type == ov::element::f16) {
    return SoftmaxMslKernelDescriptor{"metal/generated/softmax_f16",
                                      "gfx_metal_generated_softmax_f16", false};
  }
  return std::nullopt;
}

GfxMslGeneratedKernelSourcePlan
make_softmax_msl_kernel_source_plan(const std::shared_ptr<const ov::Node> &node,
                                    mlir::ModuleOp module) {
  const auto descriptor = softmax_msl_kernel_descriptor(node);
  if (!descriptor || !softmax_msl_type_supported(node)) {
    return {};
  }

  const auto axis = softmax_axis_from_node(node);
  OPENVINO_ASSERT(axis.has_value(), "GFX Metal Softmax: missing axis");
  const ov::Shape input_shape = node->get_input_shape(0);
  OPENVINO_ASSERT(!input_shape.empty(),
                  "GFX Metal Softmax: input tensor shape is unknown");

  const auto dims = compute_softmax_dims(input_shape, *axis, "GFX Metal");
  const std::vector<int32_t> scalar_args{static_cast<int32_t>(dims.rows),
                                         static_cast<int32_t>(dims.axis_len),
                                         static_cast<int32_t>(dims.inner)};

  auto binding = make_backend_custom_kernel_binding_plan(
      node->get_type_name(), descriptor->entry_point, scalar_args);
  if (!binding.valid) {
    return {};
  }

  const auto &kernel_source = softmax_msl_source(*descriptor);
  auto source =
      make_kernel_source(module, std::string(kernel_source.entry_point),
                         std::string(kernel_source.source),
                         /*arg_count=*/5u);
  auto plan =
      make_msl_generated_custom_kernel_source_plan(std::move(source), binding);
  plan.source.module = module;
  return plan;
}

GfxMslGeneratedKernelSourcePlan
make_softmax_runtime_params_msl_kernel_source_plan(
    const std::shared_ptr<const ov::Node> &node, mlir::ModuleOp module) {
  if (!node || !softmax_axis_from_node(node) ||
      !softmax_msl_type_supported(node)) {
    return {};
  }

  const ov::Shape input_shape = node->get_input_shape(0);
  OPENVINO_ASSERT(!input_shape.empty(),
                  "GFX Metal Softmax: input tensor shape is unknown");
  const auto dims = compute_softmax_dims(
      input_shape, *softmax_axis_from_node(node), "GFX Metal Softmax");

  SoftmaxCodegenDesc desc{};
  desc.element_type = node->get_output_element_type(0);
  desc.rows = static_cast<int64_t>(dims.rows);
  desc.cols = static_cast<int64_t>(dims.axis_len);
  desc.inner = static_cast<int64_t>(dims.inner);
  desc.log_softmax = softmax_is_log(node);

  module = make_softmax_codegen_module(node, module, input_shape);
  if (!module) {
    return {};
  }

  auto binding = make_backend_custom_kernel_roles_binding_plan(
      node->get_type_name(), "softmax_kernel",
      {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
       GfxKernelBufferRole::RuntimeParams});
  if (!binding.valid) {
    return {};
  }

  auto source = make_kernel_source(module, "softmax_kernel",
                                   generate_msl_from_mlir(module, desc),
                                   /*arg_count=*/3u);
  auto plan =
      make_msl_generated_custom_kernel_source_plan(std::move(source), binding);
  plan.source.module = module;
  return plan;
}

std::optional<KernelSource> make_apple_metal_softmax_kernel_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node,
    const std::optional<ov::Shape> &runtime_input_shape) {
  if (auto binding = existing_softmax_runtime_params_binding(source)) {
    if (auto configured = make_runtime_params_softmax_source(
            std::move(source), node, runtime_input_shape, *binding)) {
      return configured;
    }
    return std::nullopt;
  }

  auto plan = make_softmax_msl_kernel_source_plan(node, source.module);
  if (!plan.valid()) {
    return std::nullopt;
  }
  plan.source.mpsrt_const_tensors = std::move(source.mpsrt_const_tensors);
  return plan.source;
}

} // namespace gfx_plugin
} // namespace ov
