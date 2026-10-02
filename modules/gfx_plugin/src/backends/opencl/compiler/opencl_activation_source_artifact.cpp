// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/compiler/opencl_activation_kernel_unit.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "openvino/core/shape_util.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/acos.hpp"
#include "openvino/op/acosh.hpp"
#include "openvino/op/asin.hpp"
#include "openvino/op/asinh.hpp"
#include "openvino/op/atan.hpp"
#include "openvino/op/atanh.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/cosh.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/erf.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/hsigmoid.hpp"
#include "openvino/op/hswish.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/mish.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/sinh.hpp"
#include "openvino/op/softplus.hpp"
#include "openvino/op/softsign.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/tan.hpp"
#include "openvino/op/tanh.hpp"
#include "kernel_ir/opencl_kernels/activation_kernel.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

bool is_f32_tensor_type(const ov::element::Type &type) {
  return type == ov::element::f32;
}

bool is_f16_tensor_type(const ov::element::Type &type) {
  return type == ov::element::f16;
}

const char *opencl_scalar_type_suffix(const ov::element::Type &type) {
  if (is_f16_tensor_type(type)) {
    return "f16";
  }
  if (is_f32_tensor_type(type)) {
    return "f32";
  }
  return "unknown";
}

bool input_static_element_count_matches_output(
    const std::shared_ptr<const ov::Node> &node, size_t input_idx,
    size_t output_idx) {
  if (!node || input_idx >= node->get_input_size() ||
      output_idx >= node->get_output_size()) {
    return false;
  }
  if (!node->get_input_partial_shape(input_idx).is_static() ||
      !node->get_output_partial_shape(output_idx).is_static()) {
    return false;
  }
  return ov::shape_size(node->get_input_shape(input_idx)) ==
         ov::shape_size(node->get_output_shape(output_idx));
}

bool scalar_float_input(const std::shared_ptr<const ov::Node> &node,
                        size_t input_idx) {
  if (!node || input_idx >= node->get_input_size()) {
    return false;
  }
  return node->get_input_element_type(input_idx) ==
             node->get_input_element_type(0) &&
         node->get_input_partial_shape(input_idx).is_static() &&
         ov::shape_size(node->get_input_shape(input_idx)) == 1;
}

std::optional<float>
scalar_float_constant_input(const std::shared_ptr<const ov::Node> &node,
                            size_t input_idx) {
  if (!node || input_idx >= node->get_input_size()) {
    return std::nullopt;
  }
  auto constant = ov::as_type_ptr<const ov::op::v0::Constant>(
      node->input_value(input_idx).get_node_shared_ptr());
  if (!constant ||
      constant->get_output_element_type(0) != node->get_input_element_type(0) ||
      !constant->get_output_partial_shape(0).is_static() ||
      ov::shape_size(constant->get_output_shape(0)) != 1) {
    return std::nullopt;
  }
  const auto values = constant->cast_vector<float>();
  if (values.empty()) {
    return std::nullopt;
  }
  return values.front();
}

std::optional<GfxOpenClArtifactOp>
activation_op_code(const std::shared_ptr<const ov::Node> &node) {
  if (ov::as_type_ptr<const ov::op::v0::Relu>(node)) {
    return GfxOpenClArtifactOp::Relu;
  }
  if (ov::as_type_ptr<const ov::op::v0::Sigmoid>(node)) {
    return GfxOpenClArtifactOp::Sigmoid;
  }
  if (ov::as_type_ptr<const ov::op::v0::Tanh>(node)) {
    return GfxOpenClArtifactOp::Tanh;
  }
  if (ov::as_type_ptr<const ov::op::v0::Gelu>(node)) {
    return GfxOpenClArtifactOp::GeluErf;
  }
  if (auto gelu = ov::as_type_ptr<const ov::op::v7::Gelu>(node)) {
    return gelu->get_approximation_mode() == ov::op::GeluApproximationMode::TANH
               ? GfxOpenClArtifactOp::GeluTanh
               : GfxOpenClArtifactOp::GeluErf;
  }
  if (ov::as_type_ptr<const ov::op::v4::HSwish>(node)) {
    return GfxOpenClArtifactOp::HSwish;
  }
  if (ov::as_type_ptr<const ov::op::v5::HSigmoid>(node)) {
    return GfxOpenClArtifactOp::HSigmoid;
  }
  if (ov::as_type_ptr<const ov::op::v4::SoftPlus>(node)) {
    return GfxOpenClArtifactOp::SoftPlus;
  }
  if (ov::as_type_ptr<const ov::op::v4::Swish>(node)) {
    return GfxOpenClArtifactOp::Swish;
  }
  if (ov::as_type_ptr<const ov::op::v4::Mish>(node)) {
    return GfxOpenClArtifactOp::Mish;
  }
  if (ov::as_type_ptr<const ov::op::v9::SoftSign>(node)) {
    return GfxOpenClArtifactOp::SoftSign;
  }
  if (ov::as_type_ptr<const ov::op::v0::Abs>(node)) {
    return GfxOpenClArtifactOp::Abs;
  }
  if (ov::as_type_ptr<const ov::op::v0::Sign>(node)) {
    return GfxOpenClArtifactOp::Sign;
  }
  if (ov::as_type_ptr<const ov::op::v0::Clamp>(node)) {
    return GfxOpenClArtifactOp::Clamp;
  }
  if (ov::as_type_ptr<const ov::op::v0::Negative>(node)) {
    return GfxOpenClArtifactOp::Negative;
  }
  if (ov::as_type_ptr<const ov::op::v0::Sin>(node)) {
    return GfxOpenClArtifactOp::Sin;
  }
  if (ov::as_type_ptr<const ov::op::v0::Cos>(node)) {
    return GfxOpenClArtifactOp::Cos;
  }
  if (ov::as_type_ptr<const ov::op::v0::Tan>(node)) {
    return GfxOpenClArtifactOp::Tan;
  }
  if (ov::as_type_ptr<const ov::op::v0::Erf>(node)) {
    return GfxOpenClArtifactOp::Erf;
  }
  if (ov::as_type_ptr<const ov::op::v0::Asin>(node)) {
    return GfxOpenClArtifactOp::Asin;
  }
  if (ov::as_type_ptr<const ov::op::v0::Acos>(node)) {
    return GfxOpenClArtifactOp::Acos;
  }
  if (ov::as_type_ptr<const ov::op::v0::Atan>(node)) {
    return GfxOpenClArtifactOp::Atan;
  }
  if (ov::as_type_ptr<const ov::op::v3::Asinh>(node)) {
    return GfxOpenClArtifactOp::Asinh;
  }
  if (ov::as_type_ptr<const ov::op::v3::Acosh>(node)) {
    return GfxOpenClArtifactOp::Acosh;
  }
  if (ov::as_type_ptr<const ov::op::v3::Atanh>(node)) {
    return GfxOpenClArtifactOp::Atanh;
  }
  if (ov::as_type_ptr<const ov::op::v0::Sinh>(node)) {
    return GfxOpenClArtifactOp::Sinh;
  }
  if (ov::as_type_ptr<const ov::op::v0::Cosh>(node)) {
    return GfxOpenClArtifactOp::Cosh;
  }
  if (auto round = ov::as_type_ptr<const ov::op::v5::Round>(node)) {
    return round->get_mode() == ov::op::v5::Round::RoundMode::HALF_TO_EVEN
               ? GfxOpenClArtifactOp::RoundEven
               : GfxOpenClArtifactOp::RoundAway;
  }
  if (ov::as_type_ptr<const ov::op::v0::Exp>(node)) {
    return GfxOpenClArtifactOp::Exp;
  }
  if (ov::as_type_ptr<const ov::op::v0::Log>(node)) {
    return GfxOpenClArtifactOp::Log;
  }
  if (ov::as_type_ptr<const ov::op::v0::Sqrt>(node)) {
    return GfxOpenClArtifactOp::Sqrt;
  }
  if (ov::as_type_ptr<const ov::op::v0::Floor>(node)) {
    return GfxOpenClArtifactOp::Floor;
  }
  if (ov::as_type_ptr<const ov::op::v0::Ceiling>(node)) {
    return GfxOpenClArtifactOp::Ceiling;
  }
  if (ov::as_type_ptr<const ov::op::v0::Elu>(node)) {
    return GfxOpenClArtifactOp::Elu;
  }
  return std::nullopt;
}

std::optional<GfxOpenClSourceArtifact>
make_opencl_activation_artifact(const std::shared_ptr<const ov::Node> &node) {
  const auto op = activation_op_code(node);
  if (!op || !node ||
      (node->get_input_size() != 1 &&
       !(ov::as_type_ptr<const ov::op::v4::Swish>(node) &&
         node->get_input_size() == 2)) ||
      node->get_output_size() != 1 ||
      node->get_input_element_type(0) != node->get_output_element_type(0) ||
      (!is_f32_tensor_type(node->get_output_element_type(0)) &&
       !is_f16_tensor_type(node->get_output_element_type(0))) ||
      !input_static_element_count_matches_output(node, 0, 0)) {
    return std::nullopt;
  }

  const std::string type_suffix =
      opencl_scalar_type_suffix(node->get_output_element_type(0));
  const std::string entry_point =
      "gfx_opencl_generated_activation_" + type_suffix;
  float alpha = 0.0f;
  float beta = 0.0f;
  if (auto elu = ov::as_type_ptr<const ov::op::v0::Elu>(node)) {
    alpha = static_cast<float>(elu->get_alpha());
  }
  if (auto clamp = ov::as_type_ptr<const ov::op::v0::Clamp>(node)) {
    alpha = static_cast<float>(clamp->get_min());
    beta = static_cast<float>(clamp->get_max());
  }
  if (auto swish = ov::as_type_ptr<const ov::op::v4::Swish>(node)) {
    alpha = 1.0f;
    if (swish->get_input_size() == 2) {
      if (!scalar_float_input(node, 1)) {
        return std::nullopt;
      }
      const auto beta_value = scalar_float_constant_input(node, 1);
      if (!beta_value) {
        const auto runtime_entry_point =
            "gfx_opencl_generated_activation_runtime_beta_" + type_suffix;
        auto manifest = make_opencl_source_manifest(
            GfxKernelStageFamily::Activation,
            "opencl:generated:activation_runtime_beta:" +
                std::string(node->get_type_name()) + ":" + type_suffix,
            runtime_entry_point,
            /*direct_inputs=*/2,
            /*scalar_arg_count=*/2);
        return make_opencl_source_artifact(
            std::move(manifest),
            "opencl/generated/activation_runtime_beta_" + type_suffix,
            opencl_generated_activation_kernel_source().source,
            {GfxOpenClSourceScalarArg::ElementCount,
             GfxOpenClSourceScalarArg::OpCode},
            {0, 1}, *op);
      }
      alpha = *beta_value;
    }
  }
  auto manifest = make_opencl_source_manifest(
      GfxKernelStageFamily::Activation,
      "opencl:generated:activation:" + std::string(node->get_type_name()) +
          ":" + type_suffix,
      entry_point,
      /*direct_inputs=*/1,
      /*scalar_arg_count=*/4);
  return make_opencl_source_artifact(
      std::move(manifest), "opencl/generated/activation_" + type_suffix,
      opencl_generated_activation_kernel_source().source,
      {GfxOpenClSourceScalarArg::ElementCount, GfxOpenClSourceScalarArg::OpCode,
       GfxOpenClSourceScalarArg::StaticF32,
       GfxOpenClSourceScalarArg::StaticF32},
      {0}, *op, GfxOpenClArtifactInputMode::Direct, 0.0f, {},
      GfxOpenClSourceElementCountSource::Output0, {alpha, beta});
}

} // namespace

std::optional<GfxOpenClSourceArtifact> make_opencl_activation_source_artifact(
    const std::shared_ptr<const ov::Node> &node,
    std::string_view expected_source_id) {
  auto artifact = make_opencl_activation_artifact(node);
  if (!artifact || !artifact->valid) {
    return std::nullopt;
  }
  if (!expected_source_id.empty() &&
      artifact->artifact_ref.source_id != expected_source_id) {
    return std::nullopt;
  }
  return artifact;
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
