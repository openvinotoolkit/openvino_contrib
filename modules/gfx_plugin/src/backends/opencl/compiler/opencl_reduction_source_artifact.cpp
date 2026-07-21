// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/compiler/opencl_reduction_kernel_unit.hpp"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kernel_ir/opencl_kernels/reduction_f32_kernel.hpp"
#include "kernel_ir/opencl_kernels/reduction_logical_bool_kernel.hpp"
#include "openvino/core/axis_set.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

constexpr size_t kMaxOpenClReductionRank = 4;
constexpr uint32_t kOpenClReductionAxisSentinel = 4;

struct OpenClReductionSpec {
  GfxOpenClArtifactOp op = GfxOpenClArtifactOp::Identity;
  bool logical = false;
  bool keep_dims = false;
  ov::AxisSet axes;
};

bool checked_u32(uint64_t value, uint32_t &out) {
  if (value > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  out = static_cast<uint32_t>(value);
  return true;
}

bool append_padded_shape_u32(const ov::Shape &shape,
                             std::vector<uint32_t> &values) {
  if (shape.size() > kMaxOpenClReductionRank) {
    return false;
  }
  for (const auto dim : shape) {
    uint32_t value = 0;
    if (!checked_u32(dim, value)) {
      return false;
    }
    values.push_back(value);
  }
  values.insert(values.end(), kMaxOpenClReductionRank - shape.size(), 1u);
  return true;
}

template <typename ReduceOp>
std::optional<OpenClReductionSpec>
reduction_spec_as(const std::shared_ptr<const ov::Node> &node,
                  GfxOpenClArtifactOp op, bool logical) {
  auto reduce = ov::as_type_ptr<const ReduceOp>(node);
  if (!reduce || !reduce->reduction_axes_constant()) {
    return std::nullopt;
  }
  OpenClReductionSpec spec;
  spec.op = op;
  spec.logical = logical;
  spec.keep_dims = reduce->get_keep_dims();
  spec.axes = reduce->get_reduction_axes();
  return spec;
}

std::optional<OpenClReductionSpec>
reduction_spec(const std::shared_ptr<const ov::Node> &node) {
  if (auto spec = reduction_spec_as<ov::op::v1::ReduceSum>(
          node, GfxOpenClArtifactOp::ReduceSum, false)) {
    return spec;
  }
  if (auto spec = reduction_spec_as<ov::op::v1::ReduceMean>(
          node, GfxOpenClArtifactOp::ReduceMean, false)) {
    return spec;
  }
  if (auto spec = reduction_spec_as<ov::op::v1::ReduceMax>(
          node, GfxOpenClArtifactOp::ReduceMax, false)) {
    return spec;
  }
  if (auto spec = reduction_spec_as<ov::op::v1::ReduceMin>(
          node, GfxOpenClArtifactOp::ReduceMin, false)) {
    return spec;
  }
  if (auto spec = reduction_spec_as<ov::op::v1::ReduceProd>(
          node, GfxOpenClArtifactOp::ReduceProd, false)) {
    return spec;
  }
  if (auto spec = reduction_spec_as<ov::op::v4::ReduceL1>(
          node, GfxOpenClArtifactOp::ReduceL1, false)) {
    return spec;
  }
  if (auto spec = reduction_spec_as<ov::op::v4::ReduceL2>(
          node, GfxOpenClArtifactOp::ReduceL2, false)) {
    return spec;
  }
  if (auto spec = reduction_spec_as<ov::op::v1::ReduceLogicalAnd>(
          node, GfxOpenClArtifactOp::ReduceLogicalAnd, true)) {
    return spec;
  }
  if (auto spec = reduction_spec_as<ov::op::v1::ReduceLogicalOr>(
          node, GfxOpenClArtifactOp::ReduceLogicalOr, true)) {
    return spec;
  }
  return std::nullopt;
}

bool reduction_types_supported(const std::shared_ptr<const ov::Node> &node,
                               const OpenClReductionSpec &spec) {
  if (spec.logical) {
    return node->get_input_element_type(0) == ov::element::boolean &&
           node->get_output_element_type(0) == ov::element::boolean;
  }
  return node->get_input_element_type(0) == ov::element::f32 &&
         node->get_output_element_type(0) == ov::element::f32;
}

std::optional<uint32_t> reduction_mask(const OpenClReductionSpec &spec,
                                       size_t input_rank) {
  uint32_t mask = 0;
  for (const auto axis : spec.axes) {
    if (axis >= input_rank || axis >= kMaxOpenClReductionRank) {
      return std::nullopt;
    }
    mask |= (1u << axis);
  }
  return mask;
}

std::vector<uint32_t> reduction_output_axis_map(const OpenClReductionSpec &spec,
                                                size_t input_rank,
                                                size_t output_rank) {
  std::vector<uint32_t> axes(kMaxOpenClReductionRank,
                             kOpenClReductionAxisSentinel);
  size_t out_axis = 0;
  for (size_t input_axis = 0; input_axis < input_rank; ++input_axis) {
    const bool reduced = spec.axes.count(input_axis) != 0;
    if (reduced) {
      if (spec.keep_dims && out_axis < output_rank) {
        ++out_axis;
      }
      continue;
    }
    if (out_axis < axes.size()) {
      axes[out_axis] = static_cast<uint32_t>(input_axis);
    }
    ++out_axis;
  }
  return axes;
}

std::optional<std::vector<uint32_t>>
reduction_static_u32_scalars(const std::shared_ptr<const ov::Node> &node,
                             const OpenClReductionSpec &spec) {
  if (!node || node->get_input_size() != 2 || node->get_output_size() != 1 ||
      !node->get_input_partial_shape(0).is_static() ||
      !node->get_output_partial_shape(0).is_static() ||
      !reduction_types_supported(node, spec)) {
    return std::nullopt;
  }

  const auto &input_shape = node->get_input_shape(0);
  const auto &output_shape = node->get_output_shape(0);
  if (input_shape.empty() || input_shape.size() > kMaxOpenClReductionRank ||
      output_shape.size() > kMaxOpenClReductionRank ||
      ov::shape_size(input_shape) == 0 || ov::shape_size(output_shape) == 0) {
    return std::nullopt;
  }

  const auto mask = reduction_mask(spec, input_shape.size());
  if (!mask) {
    return std::nullopt;
  }

  std::vector<uint32_t> scalars;
  scalars.reserve(15);
  scalars.push_back(static_cast<uint32_t>(input_shape.size()));
  scalars.push_back(static_cast<uint32_t>(output_shape.size()));
  if (!append_padded_shape_u32(input_shape, scalars) ||
      !append_padded_shape_u32(output_shape, scalars)) {
    return std::nullopt;
  }
  scalars.push_back(*mask);
  const auto output_axes =
      reduction_output_axis_map(spec, input_shape.size(), output_shape.size());
  scalars.insert(scalars.end(), output_axes.begin(), output_axes.end());
  return scalars;
}

std::vector<GfxOpenClSourceScalarArg> reduction_static_scalar_args() {
  std::vector<GfxOpenClSourceScalarArg> args = {
      GfxOpenClSourceScalarArg::ElementCount, GfxOpenClSourceScalarArg::OpCode};
  args.insert(args.end(), 15, GfxOpenClSourceScalarArg::StaticU32);
  return args;
}

const GfxKernelSource &
reduction_kernel_source(const OpenClReductionSpec &spec) {
  if (spec.logical) {
    return opencl_generated_reduction_bool_kernel_source();
  }
  return opencl_generated_reduction_f32_kernel_source();
}

bool requested_unit_matches(std::string_view requested,
                            std::string_view actual) noexcept {
  return requested.empty() || requested == actual;
}

GfxOpenClSourceArtifact make_reduction_artifact(
    const GfxKernelSource &source, const std::shared_ptr<const ov::Node> &node,
    const OpenClReductionSpec &spec, std::vector<uint32_t> static_scalars) {
  const std::string type_suffix = spec.logical ? "bool" : "f32";
  auto manifest = make_opencl_source_manifest(
      GfxKernelStageFamily::Reduction,
      "opencl:generated:Reduction:" + std::string(node->get_type_name()) + ":" +
          type_suffix,
      source.entry_point,
      /*direct_inputs=*/1,
      /*scalar_arg_count=*/17);
  return make_opencl_source_artifact(
      std::move(manifest), source.kernel_id, std::string(source.source),
      reduction_static_scalar_args(), {0}, spec.op,
      GfxOpenClArtifactInputMode::Direct, 0.0f, std::move(static_scalars));
}

} // namespace

std::optional<GfxOpenClSourceArtifact> make_opencl_reduction_source_artifact(
    const std::shared_ptr<const ov::Node> &node,
    std::string_view expected_source_id) {
  const auto spec = reduction_spec(node);
  if (!spec) {
    return std::nullopt;
  }
  const auto &source = reduction_kernel_source(*spec);
  if (!requested_unit_matches(expected_source_id, source.kernel_id)) {
    return std::nullopt;
  }
  auto static_scalars = reduction_static_u32_scalars(node, *spec);
  if (!static_scalars) {
    return std::nullopt;
  }
  return make_reduction_artifact(source, node, *spec,
                                 std::move(*static_scalars));
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
