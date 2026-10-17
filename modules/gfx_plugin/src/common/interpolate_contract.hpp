// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "openvino/core/node.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace gfx_plugin {

struct InterpolateSemanticContract {
  bool nearest = false;
  bool align_corners = false;
  bool use_half_pixel = false;
  uint32_t nearest_mode = 0;
};

inline uint32_t interpolate_nearest_mode_code(
    ov::op::util::InterpolateBase::NearestMode mode) noexcept {
  using Base = ov::op::util::InterpolateBase;
  switch (mode) {
  case Base::NearestMode::FLOOR:
  case Base::NearestMode::ROUND_PREFER_FLOOR:
    return 1u;
  case Base::NearestMode::CEIL:
  case Base::NearestMode::ROUND_PREFER_CEIL:
    return 2u;
  case Base::NearestMode::SIMPLE:
  default:
    return 0u;
  }
}

inline std::optional<InterpolateSemanticContract>
make_interpolate_semantic_contract(const ov::op::v0::Interpolate &interpolate) {
  const auto &attrs = interpolate.get_attrs();
  const std::string mode = ov::util::to_lower(attrs.mode);

  InterpolateSemanticContract contract{};
  if (mode == "nearest") {
    contract.nearest = true;
  } else if (mode == "linear") {
    contract.nearest = false;
  } else {
    return std::nullopt;
  }

  contract.align_corners = attrs.align_corners;
  contract.use_half_pixel = !attrs.align_corners;
  contract.nearest_mode = 0u;
  return contract;
}

inline std::optional<InterpolateSemanticContract>
make_interpolate_semantic_contract(
    const ov::op::util::InterpolateBase::InterpolateAttrs &attrs) {
  using Base = ov::op::util::InterpolateBase;

  InterpolateSemanticContract contract{};
  switch (attrs.mode) {
  case Base::InterpolateMode::NEAREST:
    contract.nearest = true;
    break;
  case Base::InterpolateMode::LINEAR:
  case Base::InterpolateMode::LINEAR_ONNX:
  case Base::InterpolateMode::BILINEAR_PILLOW:
    contract.nearest = false;
    break;
  default:
    return std::nullopt;
  }

  switch (attrs.coordinate_transformation_mode) {
  case Base::CoordinateTransformMode::HALF_PIXEL:
    contract.align_corners = false;
    contract.use_half_pixel = true;
    break;
  case Base::CoordinateTransformMode::ALIGN_CORNERS:
    contract.align_corners = true;
    contract.use_half_pixel = false;
    break;
  case Base::CoordinateTransformMode::ASYMMETRIC:
    contract.align_corners = false;
    contract.use_half_pixel = false;
    break;
  default:
    return std::nullopt;
  }

  contract.nearest_mode = interpolate_nearest_mode_code(attrs.nearest_mode);
  return contract;
}

inline std::optional<InterpolateSemanticContract>
make_interpolate_semantic_contract(const std::shared_ptr<const ov::Node> &node) {
  if (auto interpolate = ov::as_type_ptr<const ov::op::v0::Interpolate>(node)) {
    return make_interpolate_semantic_contract(*interpolate);
  }
  if (auto interpolate = ov::as_type_ptr<const ov::op::v4::Interpolate>(node)) {
    return make_interpolate_semantic_contract(interpolate->get_attrs());
  }
  if (auto interpolate = ov::as_type_ptr<const ov::op::v11::Interpolate>(node)) {
    return make_interpolate_semantic_contract(interpolate->get_attrs());
  }
  return std::nullopt;
}

inline bool is_interpolate_node(const std::shared_ptr<const ov::Node> &node) {
  return ov::as_type_ptr<const ov::op::v0::Interpolate>(node) ||
         ov::as_type_ptr<const ov::op::v4::Interpolate>(node) ||
         ov::as_type_ptr<const ov::op::v11::Interpolate>(node);
}

inline bool interpolate_semantics_are_bilinear_half_pixel(
    const InterpolateSemanticContract &contract) noexcept {
  return !contract.nearest && !contract.align_corners && contract.use_half_pixel;
}

} // namespace gfx_plugin
} // namespace ov
