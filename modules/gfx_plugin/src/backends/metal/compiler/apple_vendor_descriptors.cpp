// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/compiler/apple_vendor_descriptors.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

bool all_zero(const std::vector<size_t> &values) {
  return std::all_of(values.begin(), values.end(),
                     [](size_t value) { return value == 0; });
}

int64_t normalize_axis(int64_t axis, size_t rank) {
  return axis < 0 ? axis + static_cast<int64_t>(rank) : axis;
}

bool copy_2d_spatial_attrs(const ov::Strides &strides,
                           const ov::Strides &dilations,
                           const ov::CoordinateDiff &pads_begin,
                           const ov::CoordinateDiff &pads_end,
                           uint32_t out_strides[2], uint32_t out_dilations[2],
                           uint32_t out_pads[4]) {
  if (strides.size() != 2 || dilations.size() != 2 || pads_begin.size() != 2 ||
      pads_end.size() != 2) {
    return false;
  }
  if (pads_begin[0] < 0 || pads_begin[1] < 0 || pads_end[0] < 0 ||
      pads_end[1] < 0) {
    return false;
  }
  out_strides[0] = static_cast<uint32_t>(strides[0]);
  out_strides[1] = static_cast<uint32_t>(strides[1]);
  out_dilations[0] = static_cast<uint32_t>(dilations[0]);
  out_dilations[1] = static_cast<uint32_t>(dilations[1]);
  out_pads[0] = static_cast<uint32_t>(pads_begin[0]);
  out_pads[1] = static_cast<uint32_t>(pads_begin[1]);
  out_pads[2] = static_cast<uint32_t>(pads_end[0]);
  out_pads[3] = static_cast<uint32_t>(pads_end[1]);
  return true;
}

bool copy_2d_spatial_attrs(const ov::Strides &strides,
                           const ov::Strides &dilations,
                           const ov::Shape &pads_begin,
                           const ov::Shape &pads_end, uint32_t out_strides[2],
                           uint32_t out_dilations[2], uint32_t out_pads[4]) {
  if (strides.size() != 2 || dilations.size() != 2 || pads_begin.size() != 2 ||
      pads_end.size() != 2) {
    return false;
  }
  out_strides[0] = static_cast<uint32_t>(strides[0]);
  out_strides[1] = static_cast<uint32_t>(strides[1]);
  out_dilations[0] = static_cast<uint32_t>(dilations[0]);
  out_dilations[1] = static_cast<uint32_t>(dilations[1]);
  out_pads[0] = static_cast<uint32_t>(pads_begin[0]);
  out_pads[1] = static_cast<uint32_t>(pads_begin[1]);
  out_pads[2] = static_cast<uint32_t>(pads_end[0]);
  out_pads[3] = static_cast<uint32_t>(pads_end[1]);
  return true;
}

bool is_static_nchw_spatial_resize(
    const std::shared_ptr<const ov::Node> &node) {
  if (!node || !node->get_input_partial_shape(0).is_static() ||
      !node->get_output_partial_shape(0).is_static()) {
    return false;
  }

  const auto input_shape = node->get_input_shape(0);
  const auto output_shape = node->get_output_shape(0);
  return input_shape.size() == 4 && output_shape.size() == 4 &&
         input_shape[0] == output_shape[0] &&
         input_shape[1] == output_shape[1] && input_shape[2] != 0 &&
         input_shape[3] != 0 && output_shape[2] != 0 && output_shape[3] != 0;
}

bool axes_are_spatial_nchw(std::vector<int64_t> axes) {
  if (axes.size() != 2) {
    return false;
  }
  for (auto &axis : axes) {
    if (axis < 0) {
      axis += 4;
    }
  }
  std::sort(axes.begin(), axes.end());
  return axes == std::vector<int64_t>{2, 3};
}

bool axes_are_spatial_nchw(const ov::AxisSet &axes) {
  std::vector<int64_t> values;
  values.reserve(axes.size());
  for (auto axis : axes) {
    values.push_back(static_cast<int64_t>(axis));
  }
  return axes_are_spatial_nchw(std::move(values));
}

bool constant_axes_input_is_spatial_nchw_or_absent(const ov::Node &node) {
  if (node.get_input_size() < 4) {
    return true;
  }
  const auto axes_node = node.input_value(3).get_node_shared_ptr();
  const auto axes_const =
      ov::as_type_ptr<const ov::op::v0::Constant>(axes_node);
  if (!axes_const) {
    return false;
  }
  return axes_are_spatial_nchw(axes_const->cast_vector<int64_t>());
}

bool configure_bilinear_half_pixel_resize_desc(GfxMpsrtResize2DAbiDesc &desc) {
  desc = {};
  desc.nearest = 0;
  desc.align_corners = 0;
  desc.half_pixel_centers = 1;
  return true;
}

std::vector<int64_t> gfx_shape_to_i64_vector(const ov::Shape &shape) {
  std::vector<int64_t> dims;
  dims.reserve(shape.size());
  for (const auto dim : shape) {
    dims.push_back(static_cast<int64_t>(dim));
  }
  return dims;
}

bool is_mps_matrix_tensor_type(const ov::element::Type &type) {
  return type == ov::element::f32 || type == ov::element::f16;
}

std::vector<int64_t> matmul_matrix_shape(int64_t batch, int64_t rows,
                                         int64_t columns) {
  if (batch == 1) {
    return {rows, columns};
  }
  return {batch, rows, columns};
}

bool make_mps_gemm_desc_from_matmul(const std::shared_ptr<const ov::Node> &node,
                                    GfxMpsrtGemmAbiDesc &desc,
                                    GfxMpsrtTensorDesc &lhs_desc,
                                    GfxMpsrtTensorDesc &rhs_desc,
                                    GfxMpsrtTensorDesc &output_desc) {
  desc = {};
  lhs_desc = {};
  rhs_desc = {};
  output_desc = {};

  auto matmul = ov::as_type_ptr<const ov::op::v0::MatMul>(node);
  if (!matmul || matmul->get_input_size() != 2 ||
      matmul->get_output_size() != 1 ||
      !matmul->get_input_partial_shape(0).is_static() ||
      !matmul->get_input_partial_shape(1).is_static() ||
      !matmul->get_output_partial_shape(0).is_static() ||
      !is_mps_matrix_tensor_type(matmul->get_input_element_type(0)) ||
      !is_mps_matrix_tensor_type(matmul->get_input_element_type(1)) ||
      !is_mps_matrix_tensor_type(matmul->get_output_element_type(0))) {
    return false;
  }

  const auto lhs_shape = matmul->get_input_shape(0);
  const auto rhs_shape = matmul->get_input_shape(1);
  const auto out_shape = matmul->get_output_shape(0);
  if (lhs_shape.size() < 2 || rhs_shape.size() < 2 || out_shape.size() < 2) {
    return false;
  }

  const bool transpose_lhs = matmul->get_transpose_a();
  const bool transpose_rhs = matmul->get_transpose_b();
  const size_t lhs_rank = lhs_shape.size();
  const size_t out_rank = out_shape.size();
  const int64_t m = static_cast<int64_t>(out_shape[out_rank - 2]);
  const int64_t n = static_cast<int64_t>(out_shape[out_rank - 1]);
  const int64_t k = static_cast<int64_t>(
      transpose_lhs ? lhs_shape[lhs_rank - 2] : lhs_shape[lhs_rank - 1]);
  if (m <= 0 || n <= 0 || k <= 0) {
    return false;
  }

  const auto matrix_a = static_cast<uint64_t>(m * k);
  const auto matrix_b = static_cast<uint64_t>(k * n);
  const auto matrix_output = static_cast<uint64_t>(m * n);
  if (matrix_a == 0 || matrix_b == 0 || matrix_output == 0) {
    return false;
  }
  const int64_t batch =
      static_cast<int64_t>(ov::shape_size(out_shape) / matrix_output);
  const int64_t batch_a =
      static_cast<int64_t>(ov::shape_size(lhs_shape) / matrix_a);
  const int64_t batch_b =
      static_cast<int64_t>(ov::shape_size(rhs_shape) / matrix_b);
  if (batch <= 0 || !((batch_a == batch || batch_a == 1) &&
                      (batch_b == batch || batch_b == 1))) {
    return false;
  }

  desc.transpose_lhs = transpose_lhs ? 1u : 0u;
  desc.transpose_rhs = transpose_rhs ? 1u : 0u;
  desc.alpha = 1.0f;
  desc.beta = 0.0f;

  lhs_desc = gfx_mpsrt_make_tensor_desc(
      gfx_shape_to_i64_vector(lhs_shape), matmul->get_input_element_type(0),
      GfxStageStorageKind::Matrix, GfxMpsrtTensorFlagExternalIo);
  rhs_desc = gfx_mpsrt_make_tensor_desc(
      gfx_shape_to_i64_vector(rhs_shape), matmul->get_input_element_type(1),
      GfxStageStorageKind::Matrix, GfxMpsrtTensorFlagExternalIo);
  output_desc = gfx_mpsrt_make_tensor_desc(
      matmul_matrix_shape(batch, m, n), matmul->get_output_element_type(0),
      GfxStageStorageKind::Matrix, GfxMpsrtTensorFlagExternalIo);
  return true;
}

bool configure_from_interpolate_base_attrs(
    const ov::op::util::InterpolateBase::InterpolateAttrs &attrs,
    GfxMpsrtResize2DAbiDesc &desc) {
  using Base = ov::op::util::InterpolateBase;
  if (attrs.mode == Base::InterpolateMode::NEAREST ||
      (attrs.mode != Base::InterpolateMode::LINEAR &&
       attrs.mode != Base::InterpolateMode::LINEAR_ONNX &&
       attrs.mode != Base::InterpolateMode::BILINEAR_PILLOW) ||
      attrs.coordinate_transformation_mode !=
          Base::CoordinateTransformMode::HALF_PIXEL ||
      attrs.antialias || !all_zero(attrs.pads_begin) ||
      !all_zero(attrs.pads_end)) {
    return false;
  }
  return configure_bilinear_half_pixel_resize_desc(desc);
}

} // namespace

uint32_t gfx_apple_mps_conv_fused_activation_code(ActivationKind kind) {
  switch (kind) {
  case ActivationKind::Relu:
    return 1u;
  case ActivationKind::Sigmoid:
    return 2u;
  case ActivationKind::Tanh:
    return 3u;
  case ActivationKind::Abs:
    return 10u;
  case ActivationKind::Identity:
    return 0u;
  default:
    return 0u;
  }
}

bool gfx_apple_mps_conv_supports_fused_activation(ActivationKind kind) {
  return kind == ActivationKind::Identity ||
         gfx_apple_mps_conv_fused_activation_code(kind) != 0u;
}

std::string gfx_apple_mps_canonical_conv_stage_type(
    const std::shared_ptr<const ov::Node> &node,
    std::string_view fallback_stage_type) {
  if (ov::as_type_ptr<const ov::op::v1::Convolution>(node)) {
    return "Convolution";
  }
  if (ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)) {
    return "GroupConvolution";
  }
  if (fallback_stage_type == "GroupConv2D") {
    return "GroupConvolution";
  }
  return std::string(fallback_stage_type);
}

bool gfx_apple_make_mps_conv2d_desc(const std::shared_ptr<const ov::Node> &node,
                                    GfxMpsrtConv2DAbiDesc &desc,
                                    bool has_activation,
                                    ActivationKind activation) {
  desc = {};
  if (!node || node->get_input_size() < 2 ||
      !node->get_input_partial_shape(0).is_static() ||
      !node->get_input_partial_shape(1).is_static() ||
      (has_activation &&
       !gfx_apple_mps_conv_supports_fused_activation(activation))) {
    return false;
  }

  bool ok = false;
  if (auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(node)) {
    const auto input_shape = conv->get_input_shape(0);
    const auto weights_shape = conv->get_input_shape(1);
    if (input_shape.size() == 4 && weights_shape.size() == 4 &&
        weights_shape[1] != 0 && input_shape[1] % weights_shape[1] == 0) {
      desc.groups = static_cast<uint32_t>(input_shape[1] / weights_shape[1]);
      ok = copy_2d_spatial_attrs(conv->get_strides(), conv->get_dilations(),
                                 conv->get_pads_begin(), conv->get_pads_end(),
                                 desc.strides, desc.dilations, desc.pads);
    }
  } else if (auto group_conv =
                 ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)) {
    const auto input_shape = group_conv->get_input_shape(0);
    const auto weights_shape = group_conv->get_input_shape(1);
    if (input_shape.size() == 4 && weights_shape.size() == 5 &&
        weights_shape[0] != 0) {
      desc.groups = static_cast<uint32_t>(weights_shape[0]);
      ok = copy_2d_spatial_attrs(
          group_conv->get_strides(), group_conv->get_dilations(),
          group_conv->get_pads_begin(), group_conv->get_pads_end(),
          desc.strides, desc.dilations, desc.pads);
    }
  }

  if (!ok) {
    desc = {};
    return false;
  }
  if (has_activation) {
    desc.fused_activation =
        gfx_apple_mps_conv_fused_activation_code(activation);
  }
  return true;
}

bool gfx_apple_make_mps_conv2d_contract(
    const std::shared_ptr<const ov::Node> &node, bool has_bias,
    const BiasParams *bias_params, bool has_activation,
    ActivationKind activation, GfxAppleMpsVendorPrimitiveContract &contract) {
  contract = {};
  if (!gfx_apple_make_mps_conv2d_desc(node, contract.descriptor.conv2d,
                                      has_activation, activation)) {
    return false;
  }
  contract.descriptor.kind = GfxAppleMpsVendorPrimitiveKind::Conv2D;
  contract.semantic_input_roles = {GfxKernelBufferRole::TensorInput,
                                   GfxKernelBufferRole::ConstTensor};
  std::vector<GfxMpsrtTensorDesc> inputs;
  std::vector<GfxMpsrtTensorDesc> outputs;
  if (!gfx_apple_make_mps_io_tensor_descs_for_node(
          node, GfxStageStorageKind::Image, inputs, outputs) ||
      node->get_input_size() < 2 ||
      !node->get_input_partial_shape(1).is_static()) {
    contract = {};
    return false;
  }
  inputs.push_back(gfx_mpsrt_make_tensor_desc(
      gfx_shape_to_i64_vector(node->get_input_shape(1)),
      node->get_input_element_type(1), GfxStageStorageKind::Buffer,
      GfxMpsrtTensorFlagConst));
  if (has_bias) {
    if (!bias_params || bias_params->empty()) {
      contract = {};
      return false;
    }
    contract.semantic_input_roles.push_back(GfxKernelBufferRole::ConstTensor);
    const auto bias_type = bias_params->element_type == ov::element::dynamic
                               ? node->get_output_element_type(0)
                               : bias_params->element_type;
    inputs.push_back(gfx_mpsrt_make_tensor_desc(
        {static_cast<int64_t>(bias_params->values.size())}, bias_type,
        GfxStageStorageKind::Buffer, GfxMpsrtTensorFlagConst));
  }
  contract.input_descs = std::move(inputs);
  contract.output_descs = std::move(outputs);
  std::vector<GfxMpsrtExternalBufferRole> roles = {
      GfxMpsrtExternalBufferRole::TensorInput,
      GfxMpsrtExternalBufferRole::ConstBuffer,
  };
  if (has_bias) {
    roles.push_back(GfxMpsrtExternalBufferRole::ConstBuffer);
  }
  roles.push_back(GfxMpsrtExternalBufferRole::TensorOutput);
  contract.external_buffer_abi =
      gfx_mpsrt_make_external_buffer_abi_from_roles(std::move(roles));
  contract.valid = contract.external_buffer_abi.valid;
  if (!contract.valid) {
    contract = {};
  }
  return contract.valid;
}

bool gfx_apple_make_mps_gemm_contract(
    const GfxMpsrtGemmAbiDesc &desc, const GfxMpsrtTensorDesc &lhs,
    const GfxMpsrtTensorDesc &rhs, const GfxMpsrtTensorDesc &output,
    GfxAppleMpsVendorPrimitiveContract &contract) {
  contract = {};
  contract.descriptor.kind = GfxAppleMpsVendorPrimitiveKind::Gemm;
  contract.descriptor.gemm = desc;
  contract.semantic_input_roles = {GfxKernelBufferRole::TensorInput,
                                   GfxKernelBufferRole::TensorInput};
  contract.input_descs = {lhs, rhs};
  contract.output_descs = {output};
  contract.external_buffer_abi = gfx_mpsrt_make_external_buffer_abi_from_roles(
      {GfxMpsrtExternalBufferRole::TensorInput,
       GfxMpsrtExternalBufferRole::TensorInput,
       GfxMpsrtExternalBufferRole::TensorOutput});
  contract.valid = contract.external_buffer_abi.valid;
  if (!contract.valid) {
    contract = {};
  }
  return contract.valid;
}

bool gfx_apple_make_mps_gemm_contract(
    const std::shared_ptr<const ov::Node> &node,
    GfxAppleMpsVendorPrimitiveContract &contract) {
  contract = {};
  GfxMpsrtGemmAbiDesc desc{};
  GfxMpsrtTensorDesc lhs_desc{};
  GfxMpsrtTensorDesc rhs_desc{};
  GfxMpsrtTensorDesc output_desc{};
  if (!make_mps_gemm_desc_from_matmul(node, desc, lhs_desc, rhs_desc,
                                      output_desc)) {
    return false;
  }
  return gfx_apple_make_mps_gemm_contract(desc, lhs_desc, rhs_desc, output_desc,
                                          contract);
}

bool gfx_apple_make_mps_pool2d_desc(const std::shared_ptr<const ov::Node> &node,
                                    GfxMpsrtPool2DAbiDesc &desc) {
  desc = {};
  if (!node || !node->get_input_partial_shape(0).is_static() ||
      !node->get_output_partial_shape(0).is_static()) {
    return false;
  }
  const auto input_shape = node->get_input_shape(0);
  const auto output_shape = node->get_output_shape(0);
  if (input_shape.size() != 4 || output_shape.size() != 4) {
    return false;
  }

  if (auto maxpool =
          std::dynamic_pointer_cast<const ov::op::util::MaxPoolBase>(node)) {
    ov::Strides dilations(maxpool->get_kernel().size(), 1);
    if (auto p = std::dynamic_pointer_cast<const ov::op::v8::MaxPool>(node)) {
      dilations = p->get_dilations();
    } else if (auto p = std::dynamic_pointer_cast<const ov::op::v14::MaxPool>(
                   node)) {
      dilations = p->get_dilations();
    }
    if (maxpool->get_kernel().size() != 2 ||
        !copy_2d_spatial_attrs(
            maxpool->get_strides(), dilations, maxpool->get_pads_begin(),
            maxpool->get_pads_end(), desc.strides, desc.dilations, desc.pads)) {
      return false;
    }
    desc.is_avg = 0;
    desc.kernel[0] = static_cast<uint32_t>(maxpool->get_kernel()[0]);
    desc.kernel[1] = static_cast<uint32_t>(maxpool->get_kernel()[1]);
    desc.exclude_pad = 1;
    return true;
  }

  if (auto avgpool =
          std::dynamic_pointer_cast<const ov::op::util::AvgPoolBase>(node)) {
    ov::Strides dilations(avgpool->get_kernel().size(), 1);
    if (auto p = std::dynamic_pointer_cast<const ov::op::v16::AvgPool>(node)) {
      dilations = p->get_dilations();
    }
    if (avgpool->get_kernel().size() != 2 ||
        !copy_2d_spatial_attrs(
            avgpool->get_strides(), dilations, avgpool->get_pads_begin(),
            avgpool->get_pads_end(), desc.strides, desc.dilations, desc.pads)) {
      return false;
    }
    desc.is_avg = 1;
    desc.kernel[0] = static_cast<uint32_t>(avgpool->get_kernel()[0]);
    desc.kernel[1] = static_cast<uint32_t>(avgpool->get_kernel()[1]);
    desc.exclude_pad = avgpool->get_exclude_pad() ? 1u : 0u;
    return true;
  }

  return false;
}

bool gfx_apple_make_mps_resize2d_desc(
    const std::shared_ptr<const ov::Node> &node,
    GfxMpsrtResize2DAbiDesc &desc) {
  if (!is_static_nchw_spatial_resize(node)) {
    return false;
  }

  if (auto interp =
          std::dynamic_pointer_cast<const ov::op::v0::Interpolate>(node)) {
    const auto mode = ov::util::to_lower(interp->get_attrs().mode);
    if (mode != "linear" || interp->get_attrs().align_corners ||
        interp->get_attrs().antialias ||
        !all_zero(interp->get_attrs().pads_begin) ||
        !all_zero(interp->get_attrs().pads_end) ||
        !axes_are_spatial_nchw(interp->get_attrs().axes)) {
      return false;
    }
    return configure_bilinear_half_pixel_resize_desc(desc);
  }

  if (auto interp =
          std::dynamic_pointer_cast<const ov::op::v4::Interpolate>(node)) {
    if (!constant_axes_input_is_spatial_nchw_or_absent(*interp)) {
      return false;
    }
    return configure_from_interpolate_base_attrs(interp->get_attrs(), desc);
  }

  if (auto interp =
          std::dynamic_pointer_cast<const ov::op::v11::Interpolate>(node)) {
    if (!constant_axes_input_is_spatial_nchw_or_absent(*interp)) {
      return false;
    }
    return configure_from_interpolate_base_attrs(interp->get_attrs(), desc);
  }

  return false;
}

bool gfx_apple_make_mps_softmax_desc(
    const std::shared_ptr<const ov::Node> &node, GfxMpsrtSoftmaxAbiDesc &desc) {
  desc = {};
  if (!node || !node->get_input_partial_shape(0).is_static() ||
      !node->get_output_partial_shape(0).is_static()) {
    return false;
  }
  const auto input_shape = node->get_input_shape(0);
  if (input_shape.empty()) {
    return false;
  }

  int64_t axis = -1;
  if (auto sm1 = ov::as_type_ptr<const ov::op::v1::Softmax>(node)) {
    axis = sm1->get_axis();
  } else if (auto sm8 = ov::as_type_ptr<const ov::op::v8::Softmax>(node)) {
    axis = sm8->get_axis();
  } else if (ov::as_type_ptr<const ov::op::v5::LogSoftmax>(node)) {
    return false;
  } else {
    return false;
  }

  axis = normalize_axis(axis, input_shape.size());
  if (axis < 0 || axis != static_cast<int64_t>(input_shape.size() - 1)) {
    return false;
  }
  desc.axis = static_cast<uint32_t>(axis);
  desc.log_softmax = 0;
  return true;
}

bool gfx_apple_make_mps_topk_desc(const std::shared_ptr<const ov::Node> &node,
                                  GfxMpsrtTopKAbiDesc &desc) {
  desc = {};
  auto topk = ov::as_type_ptr<const ov::op::util::TopKBase>(node);
  if (!topk || !topk->get_input_partial_shape(0).is_static() ||
      !topk->get_output_partial_shape(0).is_static() ||
      !topk->get_output_partial_shape(1).is_static()) {
    return false;
  }

  const auto input_shape = topk->get_input_shape(0);
  if (input_shape.empty()) {
    return false;
  }
  const int64_t axis = normalize_axis(topk->get_axis(), input_shape.size());
  if (axis < 0 || axis != static_cast<int64_t>(input_shape.size() - 1)) {
    return false;
  }
  const auto k = topk->get_k();
  if (k == 0 || k > input_shape[static_cast<size_t>(axis)] ||
      topk->get_mode() != ov::op::TopKMode::MAX) {
    return false;
  }
  const auto index_type = topk->get_output_element_type(1);
  if (index_type != ov::element::i32 && index_type != ov::element::u32 &&
      index_type != ov::element::i64) {
    return false;
  }

  desc.axis = static_cast<uint32_t>(axis);
  desc.k = static_cast<uint32_t>(k);
  desc.mode_max = 1;
  switch (topk->get_sort_type()) {
  case ov::op::TopKSortType::SORT_INDICES:
    if (k > 16) {
      return false;
    }
    desc.sort_type = 2u;
    break;
  case ov::op::TopKSortType::NONE:
    desc.sort_type = 0u;
    break;
  case ov::op::TopKSortType::SORT_VALUES:
  default:
    desc.sort_type = 1u;
    break;
  }
  return true;
}

bool gfx_apple_make_mps_sdpa_desc(const std::shared_ptr<const ov::Node> &node,
                                  GfxMpsrtSdpaAbiDesc &desc) {
  desc = {};
  auto sdpa =
      ov::as_type_ptr<const ov::op::v13::ScaledDotProductAttention>(node);
  if (!sdpa || sdpa->get_input_size() != 3 || sdpa->get_causal() ||
      !sdpa->get_input_partial_shape(0).is_static() ||
      !sdpa->get_input_partial_shape(1).is_static() ||
      !sdpa->get_input_partial_shape(2).is_static() ||
      !sdpa->get_output_partial_shape(0).is_static()) {
    return false;
  }
  const auto q_shape = sdpa->get_input_shape(0);
  const auto k_shape = sdpa->get_input_shape(1);
  const auto v_shape = sdpa->get_input_shape(2);
  const auto out_shape = sdpa->get_output_shape(0);
  if (q_shape.size() != 4 || k_shape.size() != 4 || v_shape.size() != 4 ||
      out_shape.size() != 4 || q_shape[0] != k_shape[0] ||
      q_shape[0] != v_shape[0] || q_shape[1] != k_shape[1] ||
      q_shape[1] != v_shape[1] || q_shape[3] != k_shape[3] ||
      q_shape[3] != v_shape[3] || k_shape[2] != v_shape[2] ||
      out_shape[0] != q_shape[0] || out_shape[1] != q_shape[1] ||
      out_shape[2] != q_shape[2] || out_shape[3] != v_shape[3]) {
    return false;
  }
  desc.has_mask = 0;
  desc.causal = 0;
  desc.accumulate_fp32 = 1;
  desc.scale = 1.0f / std::sqrt(static_cast<float>(q_shape[3]));
  return true;
}

bool gfx_apple_make_mps_io_tensor_descs_for_node(
    const std::shared_ptr<const ov::Node> &node, GfxStageStorageKind storage,
    std::vector<GfxMpsrtTensorDesc> &inputs,
    std::vector<GfxMpsrtTensorDesc> &outputs) {
  inputs.clear();
  outputs.clear();
  if (!node || node->get_input_size() == 0 || node->get_output_size() == 0 ||
      !node->get_input_partial_shape(0).is_static()) {
    return false;
  }

  inputs.push_back(gfx_mpsrt_make_tensor_desc(
      gfx_shape_to_i64_vector(node->get_input_shape(0)),
      node->get_input_element_type(0), storage, GfxMpsrtTensorFlagExternalIo));
  outputs.reserve(node->get_output_size());
  for (size_t output_index = 0; output_index < node->get_output_size();
       ++output_index) {
    if (!node->get_output_partial_shape(output_index).is_static()) {
      inputs.clear();
      outputs.clear();
      return false;
    }
    outputs.push_back(gfx_mpsrt_make_tensor_desc(
        gfx_shape_to_i64_vector(node->get_output_shape(output_index)),
        node->get_output_element_type(output_index), storage,
        GfxMpsrtTensorFlagTransient));
  }
  return true;
}

bool gfx_apple_make_mps_pool2d_contract(
    const std::shared_ptr<const ov::Node> &node,
    const GfxMpsrtPool2DAbiDesc &desc,
    GfxAppleMpsVendorPrimitiveContract &contract) {
  contract = {};
  contract.descriptor.kind = GfxAppleMpsVendorPrimitiveKind::Pool2D;
  contract.descriptor.pool2d = desc;
  if (!node || node->get_output_size() != 1) {
    contract = {};
    return false;
  }
  if (!gfx_apple_make_mps_io_tensor_descs_for_node(
          node, GfxStageStorageKind::Image, contract.input_descs,
          contract.output_descs)) {
    contract = {};
    return false;
  }
  contract.external_buffer_abi = gfx_mpsrt_make_external_buffer_abi_from_roles(
      {GfxMpsrtExternalBufferRole::TensorInput,
       GfxMpsrtExternalBufferRole::RuntimeParams,
       GfxMpsrtExternalBufferRole::TensorOutput});
  contract.valid = contract.external_buffer_abi.valid;
  return contract.valid;
}

bool gfx_apple_make_mps_resize2d_contract(
    const std::shared_ptr<const ov::Node> &node,
    const GfxMpsrtResize2DAbiDesc &desc,
    GfxAppleMpsVendorPrimitiveContract &contract) {
  contract = {};
  contract.descriptor.kind = GfxAppleMpsVendorPrimitiveKind::Resize2D;
  contract.descriptor.resize2d = desc;
  if (!gfx_apple_make_mps_io_tensor_descs_for_node(
          node, GfxStageStorageKind::Image, contract.input_descs,
          contract.output_descs)) {
    contract = {};
    return false;
  }
  contract.external_buffer_abi = gfx_mpsrt_make_external_buffer_abi_from_roles(
      {GfxMpsrtExternalBufferRole::TensorInput,
       GfxMpsrtExternalBufferRole::TensorOutput});
  contract.valid = contract.external_buffer_abi.valid;
  return contract.valid;
}

bool gfx_apple_make_mps_softmax_contract(
    const std::shared_ptr<const ov::Node> &node,
    const GfxMpsrtSoftmaxAbiDesc &desc,
    GfxAppleMpsVendorPrimitiveContract &contract) {
  contract = {};
  contract.descriptor.kind = GfxAppleMpsVendorPrimitiveKind::Softmax;
  contract.descriptor.softmax = desc;
  if (!gfx_apple_make_mps_io_tensor_descs_for_node(
          node, GfxStageStorageKind::Matrix, contract.input_descs,
          contract.output_descs)) {
    contract = {};
    return false;
  }
  contract.semantic_input_roles = {GfxKernelBufferRole::TensorInput,
                                   GfxKernelBufferRole::TensorOutput,
                                   GfxKernelBufferRole::RuntimeParams};
  contract.external_buffer_abi = gfx_mpsrt_make_external_buffer_abi_from_roles(
      {GfxMpsrtExternalBufferRole::TensorInput,
       GfxMpsrtExternalBufferRole::TensorOutput});
  contract.valid = contract.external_buffer_abi.valid;
  return contract.valid;
}

bool gfx_apple_make_mps_topk_contract(
    const std::shared_ptr<const ov::Node> &node,
    const GfxMpsrtTopKAbiDesc &desc,
    GfxAppleMpsVendorPrimitiveContract &contract) {
  contract = {};
  contract.descriptor.kind = GfxAppleMpsVendorPrimitiveKind::TopK;
  contract.descriptor.topk = desc;
  if (!gfx_apple_make_mps_io_tensor_descs_for_node(
          node, GfxStageStorageKind::Matrix, contract.input_descs,
          contract.output_descs)) {
    contract = {};
    return false;
  }
  contract.semantic_input_roles = {GfxKernelBufferRole::TensorInput,
                                   GfxKernelBufferRole::TensorOutput,
                                   GfxKernelBufferRole::TensorOutput};
  contract.external_buffer_abi = gfx_mpsrt_make_external_buffer_abi_from_roles(
      {GfxMpsrtExternalBufferRole::TensorInput,
       GfxMpsrtExternalBufferRole::TensorOutput,
       GfxMpsrtExternalBufferRole::TensorOutput});
  contract.valid = contract.external_buffer_abi.valid;
  return contract.valid;
}

bool gfx_apple_make_mps_sdpa_contract(
    const std::shared_ptr<const ov::Node> &node,
    const GfxMpsrtSdpaAbiDesc &desc,
    GfxAppleMpsVendorPrimitiveContract &contract) {
  contract = {};
  contract.descriptor.kind = GfxAppleMpsVendorPrimitiveKind::Sdpa;
  contract.descriptor.sdpa = desc;
  if (!node || node->get_input_size() != 3 || node->get_output_size() != 1) {
    return false;
  }
  for (size_t input_idx = 0; input_idx < 3; ++input_idx) {
    if (!node->get_input_partial_shape(input_idx).is_static()) {
      return false;
    }
    contract.input_descs.push_back(gfx_mpsrt_make_tensor_desc(
        gfx_shape_to_i64_vector(node->get_input_shape(input_idx)),
        node->get_input_element_type(input_idx), GfxStageStorageKind::NDArray,
        GfxMpsrtTensorFlagExternalIo));
  }
  if (!node->get_output_partial_shape(0).is_static()) {
    contract = {};
    return false;
  }
  contract.output_descs.push_back(gfx_mpsrt_make_tensor_desc(
      gfx_shape_to_i64_vector(node->get_output_shape(0)),
      node->get_output_element_type(0), GfxStageStorageKind::NDArray,
      GfxMpsrtTensorFlagTransient));
  contract.semantic_input_roles = {GfxKernelBufferRole::TensorInput,
                                   GfxKernelBufferRole::TensorInput,
                                   GfxKernelBufferRole::TensorInput};
  contract.external_buffer_abi = gfx_mpsrt_make_external_buffer_abi_from_roles(
      {GfxMpsrtExternalBufferRole::TensorInput,
       GfxMpsrtExternalBufferRole::TensorInput,
       GfxMpsrtExternalBufferRole::TensorInput,
       GfxMpsrtExternalBufferRole::TensorOutput});
  contract.valid = contract.external_buffer_abi.valid;
  return contract.valid;
}

bool gfx_apple_make_mps_transposed_sdpa_contract(
    std::string_view name, const ov::element::Type &element_type,
    const ov::Shape &query_shape, const ov::Shape &key_shape,
    const ov::Shape &value_shape, const ov::Shape &output_shape, float scale,
    GfxAppleMpsVendorPrimitiveContract &contract) {
  (void)name;
  contract = {};
  if ((element_type != ov::element::f32 && element_type != ov::element::f16) ||
      query_shape.size() != 4 || key_shape.size() != 4 ||
      value_shape.size() != 4 || output_shape.size() != 4 ||
      query_shape[0] != key_shape[0] || query_shape[0] != value_shape[0] ||
      query_shape[1] != key_shape[1] || query_shape[1] != value_shape[1] ||
      query_shape[2] != key_shape[2] || key_shape[3] != value_shape[3] ||
      output_shape[0] != query_shape[0] || output_shape[1] != query_shape[1] ||
      output_shape[2] != value_shape[2] || output_shape[3] != query_shape[3]) {
    return false;
  }

  GfxMpsrtSdpaAbiDesc desc{};
  desc.has_mask = 0;
  desc.causal = 0;
  desc.accumulate_fp32 = 1;
  desc.layout = GfxMpsrtSdpaLayoutTransposedBHDN;
  desc.scale = scale;

  contract.descriptor.kind = GfxAppleMpsVendorPrimitiveKind::Sdpa;
  contract.descriptor.sdpa = desc;
  contract.input_descs.push_back(gfx_mpsrt_make_tensor_desc(
      gfx_shape_to_i64_vector(query_shape), element_type,
      GfxStageStorageKind::NDArray, GfxMpsrtTensorFlagExternalIo));
  contract.input_descs.push_back(gfx_mpsrt_make_tensor_desc(
      gfx_shape_to_i64_vector(key_shape), element_type,
      GfxStageStorageKind::NDArray, GfxMpsrtTensorFlagExternalIo));
  contract.input_descs.push_back(gfx_mpsrt_make_tensor_desc(
      gfx_shape_to_i64_vector(value_shape), element_type,
      GfxStageStorageKind::NDArray, GfxMpsrtTensorFlagExternalIo));
  contract.output_descs.push_back(gfx_mpsrt_make_tensor_desc(
      gfx_shape_to_i64_vector(output_shape), element_type,
      GfxStageStorageKind::NDArray, GfxMpsrtTensorFlagTransient));
  contract.semantic_input_roles = {GfxKernelBufferRole::TensorInput,
                                   GfxKernelBufferRole::TensorInput,
                                   GfxKernelBufferRole::TensorInput};
  contract.external_buffer_abi = gfx_mpsrt_make_external_buffer_abi_from_roles(
      {GfxMpsrtExternalBufferRole::TensorInput,
       GfxMpsrtExternalBufferRole::TensorInput,
       GfxMpsrtExternalBufferRole::TensorInput,
       GfxMpsrtExternalBufferRole::TensorOutput});
  contract.valid = contract.external_buffer_abi.valid;
  return contract.valid;
}

} // namespace gfx_plugin
} // namespace ov
