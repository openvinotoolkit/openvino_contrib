// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/compiler/opencl_interpolate_kernel_unit.hpp"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "common/interpolate_contract.hpp"
#include "kernel_ir/opencl_kernels/interpolate_f16_kernel.hpp"
#include "kernel_ir/opencl_kernels/interpolate_f32_kernel.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/interpolate.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

bool is_interpolate_f32_type(const ov::element::Type &type) {
  return type == ov::element::f32;
}

bool is_interpolate_f16_type(const ov::element::Type &type) {
  return type == ov::element::f16;
}

const char *interpolate_type_suffix(const ov::element::Type &type) {
  return is_interpolate_f16_type(type) ? "f16" : "f32";
}

bool requested_unit_matches(std::string_view requested,
                            std::string_view actual) noexcept {
  return requested.empty() || requested == actual;
}

bool checked_u32(uint64_t value) {
  return value <= std::numeric_limits<uint32_t>::max();
}

bool opencl_interpolate_axes_supported(
    const std::shared_ptr<const ov::Node> &node) {
  if (auto interpolate = ov::as_type_ptr<const ov::op::v0::Interpolate>(node)) {
    return interpolate->get_attrs().axes == ov::AxisSet{2, 3};
  }
  return true;
}

bool static_nchw_spatial_resize_supported(
    const std::shared_ptr<const ov::Node> &node) {
  if (!node || node->get_input_size() < 1 || node->get_output_size() != 1 ||
      !node->get_input_partial_shape(0).is_static() ||
      !node->get_output_partial_shape(0).is_static()) {
    return false;
  }

  const auto &input_shape = node->get_input_shape(0);
  const auto &output_shape = node->get_output_shape(0);
  if (input_shape.size() != 4 || output_shape.size() != 4 ||
      input_shape[0] != output_shape[0] || input_shape[1] != output_shape[1] ||
      ov::shape_size(input_shape) == 0 || ov::shape_size(output_shape) == 0) {
    return false;
  }

  const auto output_elements = ov::shape_size(output_shape);
  return checked_u32(output_elements) && checked_u32(input_shape[0]) &&
         checked_u32(input_shape[1]) && checked_u32(input_shape[2]) &&
         checked_u32(input_shape[3]) && checked_u32(output_shape[2]) &&
         checked_u32(output_shape[3]);
}

std::vector<GfxOpenClSourceScalarArg> interpolate_scalar_args() {
  return {GfxOpenClSourceScalarArg::ElementCount,
          GfxOpenClSourceScalarArg::StaticU32,
          GfxOpenClSourceScalarArg::StaticU32,
          GfxOpenClSourceScalarArg::StaticU32,
          GfxOpenClSourceScalarArg::StaticU32,
          GfxOpenClSourceScalarArg::Input0Dim0,
          GfxOpenClSourceScalarArg::Input0Dim1,
          GfxOpenClSourceScalarArg::Input0Dim2,
          GfxOpenClSourceScalarArg::Input0Dim3,
          GfxOpenClSourceScalarArg::Output0Dim2,
          GfxOpenClSourceScalarArg::Output0Dim3};
}

const GfxKernelSource *
interpolate_kernel_source(const ov::element::Type &type) noexcept {
  if (is_interpolate_f32_type(type)) {
    return &opencl_generated_interpolate_f32_kernel_source();
  }
  if (is_interpolate_f16_type(type)) {
    return &opencl_generated_interpolate_f16_kernel_source();
  }
  return nullptr;
}

GfxOpenClSourceArtifact
make_interpolate_artifact(const GfxKernelSource &source,
                          std::string specialization_key,
                          InterpolateSemanticContract semantic) {
  auto scalar_args = interpolate_scalar_args();
  std::vector<uint32_t> static_u32_scalars = {
      semantic.nearest ? 1u : 0u,
      semantic.align_corners ? 1u : 0u,
      semantic.use_half_pixel ? 1u : 0u,
      semantic.nearest_mode,
  };

  return make_opencl_source_artifact(
      make_opencl_source_manifest(
          GfxKernelStageFamily::Resize, std::move(specialization_key),
          source.entry_point, 1, static_cast<uint32_t>(scalar_args.size())),
      source.kernel_id, std::string(source.source), std::move(scalar_args), {0},
      GfxOpenClArtifactOp::Interpolate, GfxOpenClArtifactInputMode::Direct,
      0.0f, std::move(static_u32_scalars));
}

} // namespace

std::optional<GfxOpenClSourceArtifact> make_opencl_interpolate_source_artifact(
    const std::shared_ptr<const ov::Node> &node,
    std::string_view requested_kernel_unit_id) {
  if (!is_opencl_interpolate_node(node) ||
      !static_nchw_spatial_resize_supported(node) ||
      !opencl_interpolate_axes_supported(node)) {
    return std::nullopt;
  }

  const auto element_type = node->get_input_element_type(0);
  if (element_type != node->get_output_element_type(0)) {
    return std::nullopt;
  }
  const auto *source = interpolate_kernel_source(element_type);
  if (!source ||
      !requested_unit_matches(requested_kernel_unit_id, source->kernel_id)) {
    return std::nullopt;
  }

  auto semantic = make_interpolate_semantic_contract(node);
  if (!semantic) {
    return std::nullopt;
  }

  return make_interpolate_artifact(
      *source,
      "opencl:generated:" + std::string(node->get_type_name()) + ":" +
          interpolate_type_suffix(element_type),
      *semantic);
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
