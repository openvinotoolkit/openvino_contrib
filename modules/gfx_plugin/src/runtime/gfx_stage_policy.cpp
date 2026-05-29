// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gfx_stage_policy.hpp"

#include <algorithm>
#include <string_view>
#include <utility>
#include <vector>

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
#include "openvino/op/transpose.hpp"
#include "runtime/gfx_compile_profiling.hpp"
#include "runtime/gfx_parallelism.hpp"
#include "runtime/gfx_shape_utils.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

bool is_identity_pointwise_conv(const std::shared_ptr<const ov::Node> &node) {
  auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(node);
  if (!conv || conv->get_input_size() != 2 || conv->get_output_size() != 1) {
    return false;
  }
  const auto &in_pshape = conv->get_input_partial_shape(0);
  const auto &w_pshape = conv->get_input_partial_shape(1);
  const auto &out_pshape = conv->get_output_partial_shape(0);
  if (!in_pshape.is_static() || !w_pshape.is_static() ||
      !out_pshape.is_static()) {
    return false;
  }
  const auto &in_shape = conv->get_input_shape(0);
  const auto &w_shape = conv->get_input_shape(1);
  const auto &out_shape = conv->get_output_shape(0);
  if (in_shape.size() != 4 || w_shape.size() != 4 || out_shape.size() != 4) {
    return false;
  }
  return w_shape[2] == 1 && w_shape[3] == 1 && conv->get_strides().at(0) == 1 &&
         conv->get_strides().at(1) == 1 && conv->get_dilations().at(0) == 1 &&
         conv->get_dilations().at(1) == 1 &&
         conv->get_pads_begin().at(0) == 0 &&
         conv->get_pads_begin().at(1) == 0 && conv->get_pads_end().at(0) == 0 &&
         conv->get_pads_end().at(1) == 0 && in_shape[2] == out_shape[2] &&
         in_shape[3] == out_shape[3];
}

bool has_mps_image_conv_channel_contract(
    const std::shared_ptr<const ov::Node> &node) {
  if (auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(node)) {
    if (!conv->get_input_partial_shape(0).is_static() ||
        !conv->get_input_partial_shape(1).is_static() ||
        !conv->get_output_partial_shape(0).is_static()) {
      return false;
    }
    const auto &in_shape = conv->get_input_shape(0);
    const auto &weights_shape = conv->get_input_shape(1);
    const auto &out_shape = conv->get_output_shape(0);
    return in_shape.size() == 4 && weights_shape.size() == 4 &&
           out_shape.size() == 4 && weights_shape[0] == out_shape[1] &&
           weights_shape[1] == in_shape[1];
  }
  if (auto group_conv =
          ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)) {
    if (!group_conv->get_input_partial_shape(0).is_static() ||
        !group_conv->get_input_partial_shape(1).is_static() ||
        !group_conv->get_output_partial_shape(0).is_static()) {
      return false;
    }
    const auto &in_shape = group_conv->get_input_shape(0);
    const auto &weights_shape = group_conv->get_input_shape(1);
    const auto &out_shape = group_conv->get_output_shape(0);
    if (in_shape.size() != 4 || weights_shape.size() != 5 ||
        out_shape.size() != 4) {
      return false;
    }
    const auto groups = weights_shape[0];
    if (groups == 0 || in_shape[1] % groups != 0 ||
        out_shape[1] % groups != 0) {
      return false;
    }
    const auto input_channels_per_group = in_shape[1] / groups;
    const auto output_channels_per_group = out_shape[1] / groups;
    return weights_shape[1] == output_channels_per_group &&
           weights_shape[2] == input_channels_per_group;
  }
  return false;
}

bool has_input_type(const std::shared_ptr<const ov::Node> &node,
                    std::string_view type_name) {
  if (!node) {
    return false;
  }
  for (const auto &input : node->input_values()) {
    const auto src = input.get_node_shared_ptr();
    if (src && src->get_type_name() == type_name) {
      return true;
    }
  }
  return false;
}

bool has_consumer_type(const std::shared_ptr<const ov::Node> &node,
                       std::string_view type_name) {
  if (!node) {
    return false;
  }
  for (const auto &output : node->outputs()) {
    for (const auto &target_input : output.get_target_inputs()) {
      const auto consumer = target_input.get_node()->shared_from_this();
      if (consumer && consumer->get_type_name() == type_name) {
        return true;
      }
    }
  }
  return false;
}

bool has_adjacent_type(const std::shared_ptr<const ov::Node> &node,
                       std::string_view type_name) {
  return has_input_type(node, type_name) || has_consumer_type(node, type_name);
}

bool has_any_adjacent_type(const std::shared_ptr<const ov::Node> &node,
                           std::initializer_list<std::string_view> type_names) {
  for (const auto type_name : type_names) {
    if (has_adjacent_type(node, type_name)) {
      return true;
    }
  }
  return false;
}

bool is_attention_score_stage(const std::shared_ptr<const ov::Node> &node) {
  if (!node) {
    return false;
  }
  const std::string_view type_name = node->get_type_name();
  if (type_name == "Softmax" || type_name == "LogSoftmax") {
    return has_input_type(node, "MatMul") || has_input_type(node, "Multiply") ||
           has_consumer_type(node, "MatMul");
  }
  if (type_name == "Multiply") {
    return has_input_type(node, "MatMul") ||
           has_consumer_type(node, "Softmax") ||
           has_consumer_type(node, "MatMul");
  }
  return false;
}

bool is_chainable_mobile_conv(const std::shared_ptr<const ov::Node> &node) {
  if (!node) {
    return false;
  }
  if (has_adjacent_type(node, "Concat")) {
    return false;
  }
  if (has_any_adjacent_type(node,
                            {"MatMul", "Softmax", "LogSoftmax", "Transpose",
                             "Reshape", "Split", "VariadicSplit"})) {
    return false;
  }
  return has_any_adjacent_type(node, {"Convolution", "GroupConvolution", "Add",
                                      "Multiply", "Relu", "Sigmoid", "Gelu"});
}

bool is_conv_chain_elementwise_stage(
    const std::shared_ptr<const ov::Node> &node) {
  if (!node) {
    return false;
  }
  if (has_any_adjacent_type(node, {"Concat", "MatMul", "Softmax", "LogSoftmax",
                                   "Transpose", "Reshape", "Split",
                                   "VariadicSplit"})) {
    return false;
  }
  return has_any_adjacent_type(node, {"Convolution", "GroupConvolution"});
}

bool is_conv_like(std::string_view stage_type) {
  return stage_type == "Convolution" || stage_type == "GroupConvolution";
}

bool has_static_output_rank(const std::shared_ptr<const ov::Node> &node,
                            size_t rank) {
  if (!node || node->get_output_size() == 0) {
    return false;
  }
  const auto &pshape = node->get_output_partial_shape(0);
  return pshape.rank().is_static() &&
         static_cast<size_t>(pshape.rank().get_length()) == rank;
}

template <typename Container> bool all_zero_values(const Container &values) {
  return std::all_of(values.begin(), values.end(),
                     [](const auto &value) { return value == 0; });
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

bool interpolate_attrs_are_mps_bilinear_half_pixel(
    const ov::op::util::InterpolateBase::InterpolateAttrs &attrs) {
  using Base = ov::op::util::InterpolateBase;
  return attrs.mode != Base::InterpolateMode::NEAREST &&
         (attrs.mode == Base::InterpolateMode::LINEAR ||
          attrs.mode == Base::InterpolateMode::LINEAR_ONNX ||
          attrs.mode == Base::InterpolateMode::BILINEAR_PILLOW) &&
         attrs.coordinate_transformation_mode ==
             Base::CoordinateTransformMode::HALF_PIXEL &&
         !attrs.antialias && all_zero_values(attrs.pads_begin) &&
         all_zero_values(attrs.pads_end);
}

bool is_mps_resize2d_candidate(const std::shared_ptr<const ov::Node> &node) {
  if (!is_static_nchw_spatial_resize(node)) {
    return false;
  }
  if (auto interp = ov::as_type_ptr<const ov::op::v0::Interpolate>(node)) {
    return interp->get_attrs().mode == "linear" &&
           !interp->get_attrs().align_corners &&
           !interp->get_attrs().antialias &&
           all_zero_values(interp->get_attrs().pads_begin) &&
           all_zero_values(interp->get_attrs().pads_end) &&
           axes_are_spatial_nchw(interp->get_attrs().axes);
  }
  if (auto interp = ov::as_type_ptr<const ov::op::v4::Interpolate>(node)) {
    return constant_axes_input_is_spatial_nchw_or_absent(*interp) &&
           interpolate_attrs_are_mps_bilinear_half_pixel(interp->get_attrs());
  }
  if (auto interp = ov::as_type_ptr<const ov::op::v11::Interpolate>(node)) {
    return constant_axes_input_is_spatial_nchw_or_absent(*interp) &&
           interpolate_attrs_are_mps_bilinear_half_pixel(interp->get_attrs());
  }
  return false;
}

bool is_last_dim_softmax(const std::shared_ptr<const ov::Node> &node) {
  if (!node || node->get_input_size() == 0) {
    return false;
  }
  const auto &pshape = node->get_input_partial_shape(0);
  if (!pshape.rank().is_static()) {
    return false;
  }
  const auto rank = pshape.rank().get_length();
  if (rank <= 0) {
    return false;
  }
  auto normalize_axis = [rank](int64_t axis) {
    return axis < 0 ? axis + rank : axis;
  };
  if (auto sm1 = ov::as_type_ptr<const ov::op::v1::Softmax>(node)) {
    return normalize_axis(sm1->get_axis()) == rank - 1;
  }
  if (auto sm8 = ov::as_type_ptr<const ov::op::v8::Softmax>(node)) {
    return normalize_axis(sm8->get_axis()) == rank - 1;
  }
  if (auto ls = ov::as_type_ptr<const ov::op::v5::LogSoftmax>(node)) {
    return normalize_axis(ls->get_axis()) == rank - 1;
  }
  return false;
}

bool is_last_dim_topk(const std::shared_ptr<const ov::Node> &node) {
  auto topk = ov::as_type_ptr<const ov::op::util::TopKBase>(node);
  if (!topk || topk->get_input_size() == 0) {
    return false;
  }
  const auto &pshape = topk->get_input_partial_shape(0);
  if (!pshape.rank().is_static()) {
    return false;
  }
  const auto rank = pshape.rank().get_length();
  if (rank <= 0) {
    return false;
  }
  if (topk->get_k() == 0) {
    return false;
  }
  const int64_t axis =
      topk->get_axis() < 0 ? topk->get_axis() + rank : topk->get_axis();
  return axis == rank - 1;
}

std::string make_placement_key(GfxStageBackendDomain domain,
                               GfxStageStorageKind storage,
                               std::string_view stage_type) {
  std::string key(gfx_stage_backend_domain_name(domain));
  key += ":";
  key += gfx_stage_storage_kind_name(storage);
  key += ":";
  key += stage_type;
  return key;
}

GfxStagePlacementPlan make_placement(GfxStageBackendDomain domain,
                                     GfxStageStorageKind storage,
                                     std::string_view stage_type,
                                     bool vendor_primitive,
                                     bool custom_kernel) {
  GfxStagePlacementPlan plan{};
  plan.domain = domain;
  plan.storage = storage;
  plan.uses_vendor_primitive = vendor_primitive;
  plan.uses_custom_kernel = custom_kernel;
  plan.specialization_key = make_placement_key(domain, storage, stage_type);
  return plan;
}

void record_stage_policy_counter(std::string_view name,
                                 std::string_view stage_type) {
  increment_compile_counter(std::string("stage_policy_") + std::string(name) +
                            "_count");
  if (!stage_type.empty()) {
    increment_compile_counter(std::string("stage_policy_") + std::string(name) +
                              "_" + std::string(stage_type) + "_count");
  }
}

bool is_mps_vendor_element_type(const ov::element::Type &element_type) {
  return element_type == ov::element::f16 || element_type == ov::element::f32;
}

bool is_mps_image_vendor_element_type(
    const ov::element::Type &element_type,
    const GfxStageRuntimeTraits & /*traits*/) {
  return element_type == ov::element::f16 || element_type == ov::element::f32;
}

bool is_mps_image_stage_type(std::string_view stage_type) {
  return stage_type == "Convolution" || stage_type == "GroupConvolution" ||
         stage_type == "MaxPool" || stage_type == "AvgPool" ||
         stage_type == "Interpolate";
}

bool stage_has_real_output(const std::shared_ptr<const ov::Node> &node) {
  if (!node) {
    return false;
  }
  for (size_t i = 0; i < node->get_output_size(); ++i) {
    if (node->get_output_element_type(i).is_real()) {
      return true;
    }
  }
  return false;
}

GfxStagePrecisionPlan
select_stage_precision_plan(const std::shared_ptr<const ov::Node> &node) {
  GfxStagePrecisionPlan plan{};
  plan.keep_fp32 =
      stage_has_real_output(node) && ov::fp16_compression_is_disabled(node);
  return plan;
}

bool is_mps_matrix_vendor_element_type(std::string_view stage_type,
                                       const ov::element::Type &element_type) {
  if (is_mps_vendor_element_type(element_type)) {
    return true;
  }
  return stage_type == "TopK" && element_type == ov::element::f32;
}

ov::element::Type
declared_stage_element_type(const std::shared_ptr<const ov::Node> &node,
                            const ov::element::Type &fallback) {
  if (node && node->get_output_size() > 0 &&
      node->get_output_element_type(0) != ov::element::dynamic) {
    return node->get_output_element_type(0);
  }
  return fallback;
}

bool is_mps_image_candidate(std::string_view stage_type,
                            const std::shared_ptr<const ov::Node> &node,
                            const ov::element::Type &element_type,
                            const GfxStageRuntimeTraits &traits) {
  if (!is_mps_image_stage_type(stage_type)) {
    return false;
  }
  const auto declared_type = declared_stage_element_type(node, element_type);
  if (stage_type == "Convolution" || stage_type == "GroupConvolution") {
    if (!is_mps_image_vendor_element_type(declared_type, traits)) {
      record_stage_policy_counter("mps_image_reject_element_type", stage_type);
      return false;
    }
    const bool accepted = has_static_output_rank(node, 4) &&
                          has_mps_image_conv_channel_contract(node);
    record_stage_policy_counter(accepted ? "mps_image_accept"
                                         : "mps_image_reject_shape_contract",
                                stage_type);
    return accepted;
  }
  if (stage_type == "MaxPool" || stage_type == "AvgPool") {
    if (!is_mps_image_vendor_element_type(declared_type, traits)) {
      record_stage_policy_counter("mps_image_reject_element_type", stage_type);
      return false;
    }
    if ((stage_type == "MaxPool" || stage_type == "AvgPool") && node &&
        node->get_output_size() != 1) {
      record_stage_policy_counter("mps_image_reject_multi_output_pool",
                                  stage_type);
      return false;
    }
    const bool accepted = has_static_output_rank(node, 4);
    record_stage_policy_counter(accepted ? "mps_image_accept"
                                         : "mps_image_reject_shape_contract",
                                stage_type);
    return accepted;
  }
  if (stage_type == "Interpolate") {
    const bool accepted = is_mps_resize2d_candidate(node);
    record_stage_policy_counter(accepted ? "mps_image_accept"
                                         : "mps_image_reject_shape_contract",
                                stage_type);
    return accepted;
  }
  return false;
}

bool is_mps_matrix_candidate(std::string_view stage_type,
                             const std::shared_ptr<const ov::Node> &node,
                             const ov::element::Type &element_type) {
  const auto declared_type = declared_stage_element_type(node, element_type);
  if (!is_mps_matrix_vendor_element_type(stage_type, declared_type)) {
    return false;
  }
  if (stage_type == "MatMul") {
    return true;
  }
  if (stage_type == "Softmax" && is_last_dim_softmax(node)) {
    return true;
  }
  if (stage_type == "TopK" && is_last_dim_topk(node)) {
    return true;
  }
  return false;
}

bool is_mps_ndarray_candidate(std::string_view stage_type,
                              const std::shared_ptr<const ov::Node> &node,
                              const ov::element::Type &element_type) {
  const auto declared_type = declared_stage_element_type(node, element_type);
  if (!is_mps_vendor_element_type(declared_type) ||
      stage_type != "ScaledDotProductAttention") {
    return false;
  }
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
  if (q_shape.size() != 4 || k_shape.size() != 4 || v_shape.size() != 4) {
    return false;
  }
  return q_shape[0] == k_shape[0] && q_shape[0] == v_shape[0] &&
         q_shape[1] == k_shape[1] && q_shape[1] == v_shape[1] &&
         q_shape[3] == k_shape[3] && q_shape[3] == v_shape[3] &&
         k_shape[2] == v_shape[2];
}

GfxStagePlacementPlan
select_stage_placement(GpuBackend backend, const std::string &stage_type,
                       const std::shared_ptr<const ov::Node> &node,
                       const ov::element::Type &element_type,
                       const GfxStageRuntimeTraits &traits) {
  if (backend == GpuBackend::OpenCL) {
    return make_placement(GfxStageBackendDomain::OpenCl,
                          GfxStageStorageKind::Buffer, stage_type,
                          /*vendor_primitive=*/false,
                          /*custom_kernel=*/true);
  }
  if (backend != GpuBackend::Metal) {
    return {};
  }

  if (is_mps_image_candidate(stage_type, node, element_type, traits)) {
    return make_placement(GfxStageBackendDomain::AppleMps,
                          GfxStageStorageKind::Image, stage_type,
                          /*vendor_primitive=*/true,
                          /*custom_kernel=*/false);
  }
  if (stage_type == "MaxPool" || stage_type == "AvgPool") {
    record_stage_policy_counter("mps_pooling_reject_no_vendor_route",
                                stage_type);
    return {};
  }
  if (is_mps_ndarray_candidate(stage_type, node, element_type)) {
    record_stage_policy_counter("mps_ndarray_accept", stage_type);
    return make_placement(GfxStageBackendDomain::AppleMps,
                          GfxStageStorageKind::NDArray, stage_type,
                          /*vendor_primitive=*/true,
                          /*custom_kernel=*/false);
  }
  if (is_mps_matrix_candidate(stage_type, node, element_type)) {
    record_stage_policy_counter("mps_matrix_accept", stage_type);
    return make_placement(GfxStageBackendDomain::AppleMps,
                          GfxStageStorageKind::Matrix, stage_type,
                          /*vendor_primitive=*/true,
                          /*custom_kernel=*/false);
  }
  if (stage_type == "MatMul" || stage_type == "Softmax" ||
      stage_type == "LogSoftmax" || stage_type == "TopK") {
    record_stage_policy_counter("mps_matrix_reject", stage_type);
  }

  return make_placement(GfxStageBackendDomain::AppleMsl,
                        GfxStageStorageKind::Buffer, stage_type,
                        /*vendor_primitive=*/false,
                        /*custom_kernel=*/true);
}

GfxStageArchetype
classify_stage_archetype(const std::string &stage_type,
                         const std::shared_ptr<const ov::Node> &node,
                         const GfxStageRuntimeTraits &traits) {
  if (stage_type == "Convolution") {
    return GfxStageArchetype::Convolution;
  }
  if (stage_type == "GroupConvolution") {
    return GfxStageArchetype::GroupConvolution;
  }
  if (stage_type == "MatMul") {
    return GfxStageArchetype::MatMul;
  }
  if (traits.unary_chunked) {
    return GfxStageArchetype::UnaryElementwise;
  }
  if (traits.binary_chunked) {
    return GfxStageArchetype::BinaryElementwise;
  }
  if (traits.softmax_chunked || stage_type == "Softmax" ||
      stage_type == "LogSoftmax") {
    return GfxStageArchetype::Reduction;
  }
  if (traits.transpose_chunked || stage_type == "Transpose" ||
      stage_type == "Reshape") {
    return GfxStageArchetype::Layout;
  }
  if (traits.split_concat_chunked || stage_type == "Split" ||
      stage_type == "VariadicSplit" || stage_type == "Concat") {
    return GfxStageArchetype::SplitConcat;
  }
  if (traits.convert_chunked || stage_type == "Convert") {
    return GfxStageArchetype::Convert;
  }
  if (node && ov::is_type<ov::op::v1::Convolution>(node)) {
    return GfxStageArchetype::Convolution;
  }
  if (node && ov::is_type<ov::op::v1::GroupConvolution>(node)) {
    return GfxStageArchetype::GroupConvolution;
  }
  return GfxStageArchetype::Unknown;
}

bool is_safe_shared_activation(ActivationKind kind) {
  switch (kind) {
  case ActivationKind::Relu:
  case ActivationKind::Sigmoid:
  case ActivationKind::Tanh:
  case ActivationKind::Elu:
  case ActivationKind::Prelu:
  case ActivationKind::Gelu:
  case ActivationKind::Swish:
  case ActivationKind::HSwish:
  case ActivationKind::HSigmoid:
  case ActivationKind::Abs:
  case ActivationKind::Sign:
    return true;
  default:
    return false;
  }
}

bool is_apple_mpsrt_conv_activation_fusion_supported(ActivationKind kind) {
  switch (kind) {
  case ActivationKind::Relu:
  case ActivationKind::Sigmoid:
  case ActivationKind::Tanh:
  case ActivationKind::Abs:
  case ActivationKind::Swish:
    return true;
  default:
    return false;
  }
}

bool is_identity_permutation(const std::shared_ptr<const ov::Node> &node) {
  auto transpose = ov::as_type_ptr<const ov::op::v1::Transpose>(node);
  if (!transpose || transpose->get_input_size() != 2) {
    return false;
  }
  auto perm_const = ov::as_type_ptr<const ov::op::v0::Constant>(
      transpose->input_value(1).get_node_shared_ptr());
  if (!perm_const) {
    return false;
  }
  const auto perm = perm_const->cast_vector<int64_t>();
  const auto &in_pshape = transpose->get_input_partial_shape(0);
  const auto &out_pshape = transpose->get_output_partial_shape(0);
  if (!in_pshape.rank().is_static() || !out_pshape.rank().is_static()) {
    return false;
  }
  const auto in_rank = static_cast<size_t>(in_pshape.rank().get_length());
  const auto out_rank = static_cast<size_t>(out_pshape.rank().get_length());
  if (perm.size() != in_rank || perm.size() != out_rank) {
    return false;
  }
  for (size_t i = 0; i < perm.size(); ++i) {
    if (perm[i] != static_cast<int64_t>(i)) {
      return false;
    }
  }
  return true;
}

uint64_t shape_elements(const ov::Shape &shape) {
  uint64_t total = 1;
  for (const auto dim : shape) {
    total *= std::max<uint64_t>(1, static_cast<uint64_t>(dim));
  }
  return total;
}

uint64_t output_elements(const std::shared_ptr<const ov::Node> &node) {
  if (!node || node->get_output_size() == 0) {
    return 0;
  }
  const auto &pshape = node->get_output_partial_shape(0);
  if (!pshape.is_static()) {
    return 0;
  }
  return shape_elements(node->get_output_shape(0));
}

uint64_t conv_reduction_chunk_count(const GfxParallelismCaps &caps,
                                    uint64_t reduction_work) {
  const uint64_t wave = std::max<uint32_t>(
      1u, std::max(caps.subgroup_size, caps.preferred_simd_width));
  const uint64_t group_threads =
      std::max<uint32_t>(1u, caps.max_total_threads_per_group);
  const uint64_t waves_per_group =
      std::max<uint64_t>(1ull, group_threads / wave);
  return std::max<uint64_t>(
      1ull, std::min(waves_per_group, (reduction_work + wave - 1) / wave));
}

uint64_t floor_power_of_two_u64(uint64_t value) {
  uint64_t out = 1;
  while ((out << 1) != 0 && (out << 1) <= value) {
    out <<= 1;
  }
  return out;
}

uint64_t conv_workgroup_reduction_lanes(const GfxParallelismCaps &caps,
                                        uint64_t reduction_work) {
  const uint64_t wave = std::max<uint32_t>(
      1u, std::max(caps.subgroup_size, caps.preferred_simd_width));
  const uint64_t max_threads =
      std::max<uint32_t>(1u, caps.max_total_threads_per_group);
  const uint64_t max_lanes = std::max<uint64_t>(
      1ull, floor_power_of_two_u64(std::min(wave, max_threads)));
  const uint64_t target_lane_work = std::max<uint64_t>(1ull, wave * 8ull);
  uint64_t lanes = 1;
  while (lanes < max_lanes && lanes * target_lane_work < reduction_work) {
    lanes <<= 1;
  }
  return std::max<uint64_t>(1ull, lanes);
}

uint64_t conv_tile_preserving_reduction_lanes(
    const GfxParallelismCaps &caps, const ConvParallelismPlan &direct_plan,
    uint64_t desired_reduction_lanes) {
  const uint64_t max_threads =
      std::max<uint32_t>(1u, caps.max_total_threads_per_group);
  const uint64_t threads_h =
      std::max<uint32_t>(1u, direct_plan.dispatch.threads_h);
  const uint64_t threads_w =
      std::max<uint32_t>(1u, direct_plan.dispatch.threads_w);
  const uint64_t output_thread_lanes =
      std::max<uint64_t>(1ull, threads_h * threads_w);
  const uint64_t tile_preserving_limit =
      std::max<uint64_t>(1ull, max_threads / output_thread_lanes);
  const uint64_t pow2_limit =
      floor_power_of_two_u64(tile_preserving_limit);
  return std::max<uint64_t>(
      1ull, std::min(std::max<uint64_t>(1ull, desired_reduction_lanes),
                     std::max<uint64_t>(1ull, pow2_limit)));
}

uint64_t conv_accumulator_element_size_bytes(
    const ov::element::Type &element_type) {
  return static_cast<uint64_t>(
      std::max<size_t>(element_type.size(), ov::element::f32.size()));
}

uint64_t div_ceil_u64(uint64_t value, uint64_t divisor) {
  return (value + divisor - 1) / divisor;
}

uint64_t conv_spatial_output_reuse_lanes(const ConvParallelismPlan &plan) {
  if (!plan.dispatch.enabled) {
    return 1;
  }
  const uint64_t tile_h = std::max<uint32_t>(1u, plan.dispatch.tile_h);
  const uint64_t tile_w = std::max<uint32_t>(1u, plan.dispatch.tile_w);
  const uint64_t threads_h = std::max<uint32_t>(1u, plan.dispatch.threads_h);
  const uint64_t threads_w = std::max<uint32_t>(1u, plan.dispatch.threads_w);
  return std::max<uint64_t>(1ull, div_ceil_u64(tile_h, threads_h) *
                                      div_ceil_u64(tile_w, threads_w));
}

struct ConvSpatialInputReusePlan {
  uint64_t lanes = 1;
  uint64_t unique_width_loads = 0;
  uint64_t saved_width_loads = 0;
};

ConvSpatialInputReusePlan
conv_spatial_input_reuse_plan(const ConvParallelismPlan &plan,
                              uint64_t kernel_w, uint64_t stride_w,
                              uint64_t dilation_w) {
  ConvSpatialInputReusePlan reuse{};
  if (!plan.dispatch.enabled || kernel_w <= 1 || stride_w == 0 ||
      dilation_w == 0) {
    return reuse;
  }

  const uint64_t tile_h = std::max<uint32_t>(1u, plan.dispatch.tile_h);
  const uint64_t tile_w = std::max<uint32_t>(1u, plan.dispatch.tile_w);
  const uint64_t threads_h = std::max<uint32_t>(1u, plan.dispatch.threads_h);
  const uint64_t threads_w = std::max<uint32_t>(1u, plan.dispatch.threads_w);
  const uint64_t micro_h = div_ceil_u64(tile_h, threads_h);
  const uint64_t micro_w = div_ceil_u64(tile_w, threads_w);
  if (micro_w <= 1) {
    return reuse;
  }

  const uint64_t lane_count = micro_h * micro_w;
  const uint64_t total_width_load_refs = lane_count * kernel_w;
  std::vector<std::pair<uint64_t, uint64_t>> unique_coords;
  unique_coords.reserve(static_cast<size_t>(total_width_load_refs));
  for (uint64_t lane = 0; lane < lane_count; ++lane) {
    const uint64_t mh = lane / micro_w;
    const uint64_t mw = lane % micro_w;
    for (uint64_t kw = 0; kw < kernel_w; ++kw) {
      const auto coord = std::make_pair(mh, mw * stride_w + kw * dilation_w);
      if (std::find(unique_coords.begin(), unique_coords.end(), coord) ==
          unique_coords.end()) {
        unique_coords.push_back(coord);
      }
    }
  }
  if (unique_coords.size() >= total_width_load_refs) {
    return reuse;
  }
  reuse.lanes = lane_count;
  reuse.unique_width_loads = unique_coords.size();
  reuse.saved_width_loads =
      total_width_load_refs - static_cast<uint64_t>(unique_coords.size());
  return reuse;
}

bool conv_needs_distributed_reduction(const GfxParallelismCaps &caps,
                                      uint64_t output_elems,
                                      uint64_t reduction_work) {
  const uint64_t wave = std::max<uint32_t>(
      1u, std::max(caps.subgroup_size, caps.preferred_simd_width));
  const uint64_t group_threads =
      std::max<uint32_t>(1u, caps.max_total_threads_per_group);
  const uint64_t waves_per_group =
      std::max<uint64_t>(1ull, group_threads / wave);
  const uint64_t lane_serial_budget = wave * waves_per_group;
  return output_elems >= group_threads && reduction_work >= lane_serial_budget;
}

void mark_conv_multi_kernel_required(
    GfxConvRoutePlan &plan, std::string family,
    const ov::element::Type &element_type, uint64_t reduction_work,
    uint64_t output_elems, uint64_t reduction_chunk_count,
    uint64_t reduction_chunk_size, uint64_t workgroup_reduction_lanes,
    uint64_t workgroup_output_lanes, uint64_t output_channel_reuse_lanes,
    uint64_t spatial_output_reuse_lanes,
    const ConvSpatialInputReusePlan &spatial_input_reuse,
    uint64_t coarse_spatial_tile_elements,
    uint64_t coarse_output_channel_block) {
  plan.algorithm.requires_multi_kernel_manifest = true;
  plan.algorithm.multi_kernel_family = std::move(family);
  plan.algorithm.reduction_work = reduction_work;
  plan.algorithm.output_elements = output_elems;
  plan.algorithm.reduction_chunk_count = reduction_chunk_count;
  plan.algorithm.reduction_chunk_size = reduction_chunk_size;
  plan.algorithm.workgroup_reduction_lanes = workgroup_reduction_lanes;
  plan.algorithm.workgroup_output_lanes = workgroup_output_lanes;
  plan.algorithm.output_channel_reuse_lanes =
      std::max<uint64_t>(1ull, output_channel_reuse_lanes);
  plan.algorithm.spatial_output_reuse_lanes =
      std::max<uint64_t>(1ull, spatial_output_reuse_lanes);
  plan.algorithm.output_reuse_lanes =
      plan.algorithm.output_channel_reuse_lanes *
      plan.algorithm.spatial_output_reuse_lanes;
  plan.algorithm.spatial_input_reuse_lanes =
      std::max<uint64_t>(1ull, spatial_input_reuse.lanes);
  plan.algorithm.spatial_input_reuse_unique_width_loads =
      spatial_input_reuse.unique_width_loads;
  plan.algorithm.spatial_input_reuse_saved_width_loads =
      spatial_input_reuse.saved_width_loads;
  plan.algorithm.intermediate_elements =
      output_elems * std::max<uint64_t>(1ull, reduction_chunk_count);
  auto &manifest = plan.algorithm.multi_kernel_manifest;
  manifest.family = plan.algorithm.multi_kernel_family;
  manifest.requires_owned_intermediates = true;
  manifest.requires_owned_launch_sequence = true;
  manifest.requires_output_reuse = plan.algorithm.output_reuse_lanes > 1ull;
  manifest.requires_spatial_input_reuse =
      plan.algorithm.spatial_input_reuse_saved_width_loads > 0ull;
  manifest.has_workgroup_local_reduction_plan =
      workgroup_reduction_lanes > 1ull && workgroup_output_lanes > 0ull;
  manifest.coarse_spatial_tile_elements =
      std::max<uint64_t>(1ull, coarse_spatial_tile_elements);
  manifest.coarse_output_channel_block =
      std::max<uint64_t>(1ull, coarse_output_channel_block);
  manifest.coarse_output_tile_elements =
      manifest.coarse_spatial_tile_elements *
      manifest.coarse_output_channel_block;
  manifest.requires_coarse_output_tile_preservation =
      manifest.has_workgroup_local_reduction_plan &&
      manifest.coarse_output_tile_elements > 1ull;
  manifest.workgroup_output_tile_deficit =
      manifest.coarse_output_tile_elements > workgroup_output_lanes
          ? manifest.coarse_output_tile_elements - workgroup_output_lanes
          : 0ull;
  manifest.workgroup_local_accumulator_elements =
      std::max<uint64_t>(1ull, workgroup_reduction_lanes) *
      std::max<uint64_t>(1ull, workgroup_output_lanes);
  manifest.workgroup_local_accumulator_bytes =
      manifest.workgroup_local_accumulator_elements *
      conv_accumulator_element_size_bytes(element_type);
  manifest.stages.clear();
  if (manifest.has_workgroup_local_reduction_plan) {
    manifest.requires_owned_intermediates = false;
    manifest.requires_owned_launch_sequence = false;
    manifest.partial_sum_elements = 0;
    manifest.reduced_accumulator_elements = 0;
    manifest.owned_intermediate_elements = 0;
    manifest.owned_intermediate_bytes = 0;
    manifest.owned_intermediate_buffer_count = 0;
    manifest.stages.push_back(
        {GfxConvMultiKernelStageKind::FinalizeOutput,
         "cooperative_direct_reduce", output_elems,
         /*writes_intermediate=*/false,
         /*writes_final_output=*/true});
  } else {
    manifest.partial_sum_elements = plan.algorithm.intermediate_elements;
    manifest.reduced_accumulator_elements = output_elems;
    manifest.owned_intermediate_elements =
        manifest.partial_sum_elements + manifest.reduced_accumulator_elements;
    manifest.owned_intermediate_bytes =
        manifest.owned_intermediate_elements *
        static_cast<uint64_t>(std::max<size_t>(1, element_type.size()));
    manifest.owned_intermediate_buffer_count = 2;
    manifest.stages.push_back({GfxConvMultiKernelStageKind::PartialReduce,
                               "partial_reduce",
                               plan.algorithm.intermediate_elements,
                               /*writes_intermediate=*/true,
                               /*writes_final_output=*/false});
    manifest.stages.push_back({GfxConvMultiKernelStageKind::ReducePartialSums,
                               "reduce_partial_sums", output_elems,
                               /*writes_intermediate=*/true,
                               /*writes_final_output=*/false});
    manifest.stages.push_back({GfxConvMultiKernelStageKind::FinalizeOutput,
                               "finalize_output", output_elems,
                               /*writes_intermediate=*/false,
                               /*writes_final_output=*/true});
  }
  manifest.launch_dispatch_count = manifest.stages.size();
  plan.algorithm.variant =
      "direct_waits_for_" + plan.algorithm.multi_kernel_family + "_manifest";
}

GfxParallelismCaps query_stage_caps(const GpuBufferManager *buffer_manager,
                                    GpuBackend backend) {
  if (buffer_manager) {
    return query_parallelism_caps(buffer_manager);
  }
  GfxParallelismCaps caps{};
  caps.backend = backend;
  caps.device_family = backend == GpuBackend::Metal ? GpuDeviceFamily::Apple
                                                    : GpuDeviceFamily::Generic;
  if (backend == GpuBackend::OpenCL) {
    caps.preferred_simd_width = 32;
    caps.subgroup_size = 32;
    caps.max_total_threads_per_group = 128;
    caps.max_threads_per_group = {128, 128, 64};
  } else {
    caps.preferred_simd_width = 32;
    caps.subgroup_size = 32;
    caps.max_total_threads_per_group = 256;
    caps.max_threads_per_group = {256, 256, 64};
    caps.supports_conv_output_channel_blocking = true;
    caps.supports_conv_channel_block_spatial_tiling = true;
  }
  return caps;
}

} // namespace

bool allow_stage_bias_fusion(GpuBackend backend,
                             const std::string &stage_type) {
  if (backend == GpuBackend::OpenCL) {
    return stage_type == "Convolution";
  }
  return is_conv_like(stage_type);
}

bool allow_stage_batchnorm_fusion(GpuBackend backend,
                                  const std::string &stage_type) {
  if (backend == GpuBackend::OpenCL) {
    return stage_type == "Convolution";
  }
  return is_conv_like(stage_type);
}

bool allow_stage_activation_fusion(GpuBackend backend,
                                   const std::string &stage_type,
                                   ActivationKind kind) {
  if (!is_safe_shared_activation(kind)) {
    return false;
  }
  if (backend == GpuBackend::OpenCL) {
    return stage_type == "Convolution" &&
           (kind == ActivationKind::Relu || kind == ActivationKind::Swish);
  }
  if (backend == GpuBackend::Metal) {
    return is_conv_like(stage_type) &&
           is_apple_mpsrt_conv_activation_fusion_supported(kind);
  }
  return is_conv_like(stage_type);
}

const char *gfx_stage_backend_domain_name(GfxStageBackendDomain domain) {
  switch (domain) {
  case GfxStageBackendDomain::AppleMps:
    return "apple_mps";
  case GfxStageBackendDomain::AppleMsl:
    return "apple_msl";
  case GfxStageBackendDomain::OpenCl:
    return "opencl";
  case GfxStageBackendDomain::Unknown:
  default:
    return "unknown";
  }
}

const char *gfx_stage_storage_kind_name(GfxStageStorageKind storage) {
  switch (storage) {
  case GfxStageStorageKind::Buffer:
    return "buffer";
  case GfxStageStorageKind::Image:
    return "image";
  case GfxStageStorageKind::Matrix:
    return "matrix";
  case GfxStageStorageKind::NDArray:
    return "ndarray";
  case GfxStageStorageKind::Alias:
    return "alias";
  case GfxStageStorageKind::Unknown:
  default:
    return "unknown";
  }
}

const char *
gfx_conv_multi_kernel_stage_kind_name(GfxConvMultiKernelStageKind kind) {
  switch (kind) {
  case GfxConvMultiKernelStageKind::PartialReduce:
    return "partial_reduce";
  case GfxConvMultiKernelStageKind::ReducePartialSums:
    return "reduce_partial_sums";
  case GfxConvMultiKernelStageKind::FinalizeOutput:
    return "finalize_output";
  case GfxConvMultiKernelStageKind::Unknown:
  default:
    return "unknown";
  }
}

GfxStagePostOpSupport
select_stage_post_op_support(GpuBackend backend, GfxStageArchetype archetype,
                             const std::string &stage_type) {
  GfxStagePostOpSupport support{};
  switch (archetype) {
  case GfxStageArchetype::Convolution:
  case GfxStageArchetype::GroupConvolution:
    support.batchnorm = allow_stage_batchnorm_fusion(backend, stage_type);
    support.bias = allow_stage_bias_fusion(backend, stage_type);
    support.activation = allow_stage_activation_fusion(backend, stage_type,
                                                       ActivationKind::Relu);
    break;
  default:
    break;
  }
  return support;
}

GfxTensorLayoutPlan
select_tensor_layout_plan(const std::string &stage_type,
                          const std::shared_ptr<const ov::Node> &node) {
  GfxTensorLayoutPlan plan{};
  if (stage_type == "ReadValue" || stage_type == "Reshape" ||
      stage_type == "Squeeze" || stage_type == "Unsqueeze") {
    plan.kind = GfxTensorLayoutKind::ViewOnly;
    plan.view_only = true;
    return plan;
  }
  if (stage_type == "Transpose" && is_identity_permutation(node)) {
    plan.kind = GfxTensorLayoutKind::ViewOnly;
    plan.view_only = true;
    return plan;
  }
  if (stage_type == "Transpose" || stage_type == "Reshape" ||
      stage_type == "Squeeze" || stage_type == "Unsqueeze") {
    plan.kind = GfxTensorLayoutKind::Materialized;
  }
  return plan;
}

GfxStageOptimizationPlan select_stage_optimization_plan(
    const GpuBufferManager *buffer_manager, GpuBackend backend,
    const std::string &stage_type, const std::shared_ptr<const ov::Node> &node,
    const ov::element::Type &element_type, bool has_bias, bool has_activation,
    bool has_batchnorm, const GfxStageRuntimeTraits &traits) {
  GfxStageOptimizationPlan plan{};
  plan.archetype = classify_stage_archetype(stage_type, node, traits);
  plan.placement =
      select_stage_placement(backend, stage_type, node, element_type, traits);
  plan.precision = select_stage_precision_plan(node);
  plan.layout = select_tensor_layout_plan(stage_type, node);
  plan.post_ops =
      select_stage_post_op_support(backend, plan.archetype, stage_type);
  plan.execution.fusion.allow_bias = plan.post_ops.bias;
  plan.execution.fusion.allow_batchnorm = plan.post_ops.batchnorm;
  plan.execution.fusion.allow_activation = plan.post_ops.activation;
  if (node) {
    plan.conv =
        select_conv_route_plan(buffer_manager, backend, node, element_type,
                               has_bias, has_activation, has_batchnorm);
  }
  const bool source_kernel_backend = backend == GpuBackend::OpenCL;
  if (!source_kernel_backend) {
    return plan;
  }

  const auto caps = query_stage_caps(buffer_manager, backend);
  const uint32_t wave = std::max<uint32_t>(
      1u, std::max(caps.subgroup_size, caps.preferred_simd_width));
  const uint64_t out_elems = output_elements(node);
  constexpr uint64_t kLargeMobileChunkedOutputElems = 262144ull;
  constexpr uint64_t kChainableConvOutputElems = 1048576ull;
  const bool constrained_source_submit =
      caps.max_total_threads_per_group <= 256u && wave < 32u;
  const uint64_t large_chunked_output_elems =
      constrained_source_submit ? (kLargeMobileChunkedOutputElems * 4ull)
                                : kLargeMobileChunkedOutputElems;
  const bool chainable_mobile_conv = is_chainable_mobile_conv(node) &&
                                     out_elems > 0 &&
                                     out_elems <= kChainableConvOutputElems;

  if (plan.archetype == GfxStageArchetype::MatMul) {
    plan.execution.submit.weight = wave >= 64 ? 10 : 8;
    // Keep MatMul inside the adaptive submit window so tightly-coupled
    // layout/split epilogues can stay in the same command buffer when the
    // budget allows it. On mobile source-kernel stacks the extra cross-submit
    // hop between producer chains and MatMul tends to be more expensive than a
    // slightly wider window.
    plan.execution.submit.isolate = false;
    return plan;
  }

  if (traits.binary_chunked && is_attention_score_stage(node)) {
    // Attention score scaling is tightly coupled to the following Softmax
    // and MatMul. On mobile source-kernel stacks, forcing these stages into
    // separate submit windows is more fragile than a slightly wider window.
    plan.execution.submit.weight = 4;
    plan.execution.submit.isolate = false;
    return plan;
  }
  if (traits.binary_chunked) {
    plan.execution.submit.weight = 8;
    plan.execution.submit.isolate = !is_conv_chain_elementwise_stage(node) &&
                                    !constrained_source_submit &&
                                    out_elems >= large_chunked_output_elems;
    return plan;
  }
  if ((traits.unary_chunked || traits.softmax_chunked) &&
      is_attention_score_stage(node)) {
    plan.execution.submit.weight = 4;
    plan.execution.submit.isolate = false;
    return plan;
  }
  if (traits.unary_chunked || traits.softmax_chunked) {
    plan.execution.submit.weight = 6;
    plan.execution.submit.isolate =
        !(traits.unary_chunked && is_conv_chain_elementwise_stage(node)) &&
        !constrained_source_submit && out_elems >= large_chunked_output_elems;
    return plan;
  }
  if ((plan.archetype == GfxStageArchetype::Convolution ||
       plan.archetype == GfxStageArchetype::GroupConvolution) &&
      plan.conv.kind == GfxConvRouteKind::None) {
    // Shared source-kernel convolution lowering is still a heavy stage on
    // mobile-class drivers and should not be mixed into wide multi-op submit
    // windows, except for safe pointwise 1x1 stages where extra submit/barrier
    // churn tends to dominate more than the kernel itself on mobile GPUs.
    plan.execution.submit.weight = wave >= 64 ? 10 : 8;
    if (plan.archetype == GfxStageArchetype::Convolution &&
        is_identity_pointwise_conv(node)) {
      // Keep shared 1x1 convolutions light enough to co-reside with the
      // following unary/binary/layout epilogue stages in one submit
      // window. Profiling on mobile source-kernel stacks shows that extra
      // cross-submit barriers are more expensive here than slightly wider
      // windows.
      plan.execution.submit.weight = 4;
      plan.execution.submit.isolate = false;
    } else if (chainable_mobile_conv) {
      plan.execution.submit.isolate = false;
    } else {
      plan.execution.submit.isolate = true;
    }
    return plan;
  }
  if (traits.transpose_chunked || traits.split_concat_chunked) {
    plan.execution.submit.weight = 4;
    if (traits.split_concat_chunked &&
        out_elems >= large_chunked_output_elems) {
      plan.execution.submit.weight = 8;
      plan.execution.submit.isolate = true;
    }
    return plan;
  }
  if (traits.convert_chunked) {
    plan.execution.submit.weight = 6;
    return plan;
  }
  return plan;
}

GfxStageExecutionPolicy
select_stage_execution_policy(const GpuBufferManager *buffer_manager,
                              GpuBackend backend, const std::string &stage_type,
                              const GfxStageRuntimeTraits &traits) {
  return select_stage_optimization_plan(buffer_manager, backend, stage_type,
                                        nullptr, ov::element::dynamic, false,
                                        false, false, traits)
      .execution;
}

GfxConvRoutePlan
select_conv_route_plan(const GpuBufferManager *buffer_manager,
                       GpuBackend backend,
                       const std::shared_ptr<const ov::Node> &node,
                       const ov::element::Type &element_type, bool has_bias,
                       bool has_activation, bool has_batchnorm) {
  GfxConvRoutePlan plan{};
  const bool source_kernel_backend = backend == GpuBackend::OpenCL;
  if (!source_kernel_backend || !node) {
    return plan;
  }
  if (element_type != ov::element::f16 && element_type != ov::element::f32) {
    return plan;
  }
  if (auto gconv = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)) {
    const auto &in_pshape = gconv->get_input_partial_shape(0);
    const auto &w_pshape = gconv->get_input_partial_shape(1);
    const auto &out_pshape = gconv->get_output_partial_shape(0);
    if (!in_pshape.is_static() || !w_pshape.is_static() ||
        !out_pshape.is_static()) {
      plan.family = GfxConvFamily::Grouped;
      plan.algorithm.kind = GfxConvAlgorithmKind::Indirect;
      return plan;
    }
    const auto &in_shape = gconv->get_input_shape(0);
    const auto &w_shape = gconv->get_input_shape(1);
    const auto &out_shape = gconv->get_output_shape(0);
    if (in_shape.size() == 4 && out_shape.size() == 4 && w_shape.size() == 5 &&
        w_shape[0] == in_shape[1] && w_shape[0] == out_shape[1] &&
        w_shape[1] == 1 && w_shape[2] == 1) {
      plan.family = GfxConvFamily::Depthwise;
      plan.algorithm.kind = GfxConvAlgorithmKind::DepthwiseDirect;
      plan.algorithm.variant = "depthwise_shared_mlir";
    } else {
      plan.family = GfxConvFamily::Grouped;
      plan.algorithm.kind = GfxConvAlgorithmKind::Indirect;
    }
    return plan;
  }

  auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(node);
  if (!conv || conv->get_input_size() != 2 || conv->get_output_size() != 1) {
    return plan;
  }
  const auto &in_pshape = conv->get_input_partial_shape(0);
  const auto &w_pshape = conv->get_input_partial_shape(1);
  const auto &out_pshape = conv->get_output_partial_shape(0);
  if (!in_pshape.is_static() || !w_pshape.is_static() ||
      !out_pshape.is_static()) {
    plan.family = GfxConvFamily::General;
    plan.algorithm.kind = GfxConvAlgorithmKind::Indirect;
    return plan;
  }
  const auto &in_shape = conv->get_input_shape(0);
  const auto &w_shape = conv->get_input_shape(1);
  const auto &out_shape = conv->get_output_shape(0);
  if (in_shape.size() != 4 || w_shape.size() != 4 || out_shape.size() != 4) {
    return plan;
  }
  const bool pointwise_1x1 =
      w_shape[2] == 1 && w_shape[3] == 1 && conv->get_strides().at(0) == 1 &&
      conv->get_strides().at(1) == 1 && conv->get_dilations().at(0) == 1 &&
      conv->get_dilations().at(1) == 1 && conv->get_pads_begin().at(0) == 0 &&
      conv->get_pads_begin().at(1) == 0 && conv->get_pads_end().at(0) == 0 &&
      conv->get_pads_end().at(1) == 0;
  const bool spatial_3x3 = w_shape[2] == 3 && w_shape[3] == 3;
  if (pointwise_1x1) {
    plan.family = GfxConvFamily::Pointwise1x1;
  } else if (spatial_3x3) {
    plan.family = GfxConvFamily::Spatial3x3;
  } else {
    plan.family = GfxConvFamily::General;
  }
  const uint64_t reduction_work =
      static_cast<uint64_t>(std::max<size_t>(1, in_shape[1])) *
      static_cast<uint64_t>(std::max<size_t>(1, w_shape[2])) *
      static_cast<uint64_t>(std::max<size_t>(1, w_shape[3]));
  const uint64_t out_elems = shape_elements(out_shape);
  const auto caps = query_stage_caps(buffer_manager, backend);
  if (conv_needs_distributed_reduction(caps, out_elems, reduction_work)) {
    const uint64_t desired_workgroup_reduction_lanes =
        conv_workgroup_reduction_lanes(caps, reduction_work);
    const bool stride2 =
        conv->get_strides().at(0) > 1 || conv->get_strides().at(1) > 1;
    const auto direct_plan = select_conv_parallelism(
        caps, out_shape,
        static_cast<uint64_t>(std::max<size_t>(1, in_shape[1])),
        static_cast<uint64_t>(std::max<size_t>(1, w_shape[0])), reduction_work,
        stride2,
        /*depthwise=*/false);
    const uint64_t output_channel_reuse_lanes =
        std::max<uint32_t>(1u, direct_plan.output_channel_block);
    const uint64_t coarse_spatial_tile_elements =
        static_cast<uint64_t>(std::max<uint32_t>(1u,
                                                 direct_plan.dispatch.tile_h)) *
        static_cast<uint64_t>(std::max<uint32_t>(1u,
                                                 direct_plan.dispatch.tile_w));
    const uint64_t coarse_output_channel_block = output_channel_reuse_lanes;
    const uint64_t coarse_output_tile_elements =
        coarse_spatial_tile_elements * coarse_output_channel_block;
    const uint64_t workgroup_reduction_lanes =
        conv_tile_preserving_reduction_lanes(
            caps, direct_plan, desired_workgroup_reduction_lanes);
    const uint64_t workgroup_output_lanes =
        std::max<uint64_t>(1ull, coarse_output_tile_elements);
    const uint64_t reduction_chunk_count =
        std::max<uint64_t>(1ull, workgroup_reduction_lanes);
    const uint64_t reduction_chunk_size =
        (reduction_work + reduction_chunk_count - 1) / reduction_chunk_count;
    uint64_t spatial_output_reuse_lanes =
        conv_spatial_output_reuse_lanes(direct_plan);
    if (caps.supports_conv_channel_block_spatial_tiling &&
        output_channel_reuse_lanes > 0ull &&
        workgroup_output_lanes > output_channel_reuse_lanes &&
        workgroup_output_lanes % output_channel_reuse_lanes == 0ull) {
      spatial_output_reuse_lanes = std::max<uint64_t>(
          spatial_output_reuse_lanes,
          workgroup_output_lanes / output_channel_reuse_lanes);
    }
    const auto spatial_input_reuse = conv_spatial_input_reuse_plan(
        direct_plan, static_cast<uint64_t>(std::max<size_t>(1, w_shape[3])),
        static_cast<uint64_t>(std::max<size_t>(1, conv->get_strides().at(1))),
        static_cast<uint64_t>(
            std::max<size_t>(1, conv->get_dilations().at(1))));
    mark_conv_multi_kernel_required(
        plan, "cooperative_direct_reduce", element_type, reduction_work,
        out_elems, reduction_chunk_count, reduction_chunk_size,
        workgroup_reduction_lanes, workgroup_output_lanes,
        output_channel_reuse_lanes, spatial_output_reuse_lanes,
        spatial_input_reuse, coarse_spatial_tile_elements,
        coarse_output_channel_block);
    record_stage_policy_counter("conv_multikernel_manifest_required",
                                node->get_type_name());
  }

  // Plain Conv2D normally stays on the shared MLIR custom-kernel path.
  // Cooperative multi-kernel reductions require a subgraph/stage manifest that
  // owns partial-sum buffers and all launches. They must not be selected as a
  // single custom-kernel stage.
  (void)has_bias;
  (void)has_activation;
  (void)has_batchnorm;
  return plan;
}

} // namespace gfx_plugin
} // namespace ov
