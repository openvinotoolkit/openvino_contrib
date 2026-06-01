// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/compiler/metal_stage_placement.hpp"

#include <algorithm>
#include <initializer_list>
#include <string_view>
#include <utility>
#include <vector>

#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "runtime/gfx_compile_profiling.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

void record_stage_policy_counter(std::string_view name,
                                 std::string_view stage_type) {
  increment_compile_counter(std::string("stage_policy_") + std::string(name) +
                            "_count");
  if (!stage_type.empty()) {
    increment_compile_counter(std::string("stage_policy_") + std::string(name) +
                              "_" + std::string(stage_type) + "_count");
  }
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
    if (node && node->get_output_size() != 1) {
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

bool is_mps_matrix_vendor_element_type(std::string_view stage_type,
                                       const ov::element::Type &element_type) {
  if (is_mps_vendor_element_type(element_type)) {
    return true;
  }
  return stage_type == "TopK" && element_type == ov::element::f32;
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

class MetalStagePlacementPolicy final : public StagePlacementPolicy {
public:
  GfxStagePlacementPlan
  select_placement(const StagePlacementQuery &query) const override {
    if (is_mps_image_candidate(query.stage_type, query.node,
                               query.element_type, query.traits)) {
      return make_stage_placement(GfxStageBackendDomain::AppleMps,
                                  GfxStageStorageKind::Image, query.stage_type,
                                  /*vendor_primitive=*/true,
                                  /*custom_kernel=*/false);
    }
    if (query.stage_type == "MaxPool" || query.stage_type == "AvgPool") {
      record_stage_policy_counter("mps_pooling_reject_no_vendor_route",
                                  query.stage_type);
      return {};
    }
    if (is_mps_ndarray_candidate(query.stage_type, query.node,
                                 query.element_type)) {
      record_stage_policy_counter("mps_ndarray_accept", query.stage_type);
      return make_stage_placement(GfxStageBackendDomain::AppleMps,
                                  GfxStageStorageKind::NDArray,
                                  query.stage_type,
                                  /*vendor_primitive=*/true,
                                  /*custom_kernel=*/false);
    }
    if (is_mps_matrix_candidate(query.stage_type, query.node,
                                query.element_type)) {
      record_stage_policy_counter("mps_matrix_accept", query.stage_type);
      return make_stage_placement(GfxStageBackendDomain::AppleMps,
                                  GfxStageStorageKind::Matrix,
                                  query.stage_type,
                                  /*vendor_primitive=*/true,
                                  /*custom_kernel=*/false);
    }
    if (query.stage_type == "MatMul" || query.stage_type == "Softmax" ||
        query.stage_type == "LogSoftmax" || query.stage_type == "TopK") {
      record_stage_policy_counter("mps_matrix_reject", query.stage_type);
    }
    return make_stage_placement(GfxStageBackendDomain::AppleMsl,
                                GfxStageStorageKind::Buffer, query.stage_type,
                                /*vendor_primitive=*/false,
                                /*custom_kernel=*/true);
  }
};

} // namespace

std::shared_ptr<const StagePlacementPolicy>
make_metal_stage_placement_policy() {
  static const auto policy = std::make_shared<MetalStagePlacementPolicy>();
  return policy;
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
