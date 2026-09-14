// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/compiler/opencl_tile_kernel_unit.hpp"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kernel_ir/gfx_custom_kernel_families.hpp"
#include "kernel_ir/opencl_kernels/tile_kernel.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/tile.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

bool is_f32_tile_type(const ov::element::Type &type) {
  return type == ov::element::f32;
}

bool is_f16_tile_type(const ov::element::Type &type) {
  return type == ov::element::f16;
}

bool checked_u32(uint64_t value, uint32_t &out) {
  if (value > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  out = static_cast<uint32_t>(value);
  return true;
}

uint64_t shape_product_range(const ov::Shape &shape, size_t begin, size_t end) {
  uint64_t product = 1;
  for (size_t axis = begin; axis < end; ++axis) {
    product *= shape[axis];
  }
  return product;
}

bool append_shape_u32(const ov::Shape &shape, size_t max_rank,
                      std::vector<uint32_t> &values) {
  if (shape.size() > max_rank) {
    return false;
  }
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    uint32_t dim = 0;
    if (!checked_u32(shape[axis], dim)) {
      return false;
    }
    values.push_back(dim);
  }
  values.insert(values.end(), max_rank - shape.size(), 1u);
  return true;
}

bool append_strides_u32(const ov::Shape &shape, size_t max_rank,
                        std::vector<uint32_t> &values) {
  if (shape.size() > max_rank) {
    return false;
  }
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    uint32_t stride = 0;
    if (!checked_u32(shape_product_range(shape, axis + 1, shape.size()),
                     stride)) {
      return false;
    }
    values.push_back(stride);
  }
  values.insert(values.end(), max_rank - shape.size(), 1u);
  return true;
}

std::optional<std::vector<int64_t>>
constant_i64_vector_input(const std::shared_ptr<const ov::Node> &node,
                          size_t input_idx) {
  if (!node || input_idx >= node->get_input_size()) {
    return std::nullopt;
  }
  const auto constant = ov::as_type_ptr<const ov::op::v0::Constant>(
      node->input_value(input_idx).get_node_shared_ptr());
  if (!constant || constant->get_element_type() != ov::element::i64) {
    return std::nullopt;
  }
  return constant->cast_vector<int64_t>();
}

GfxOpenClSourceScalarArg input0_dim_scalar_arg(size_t axis) {
  return static_cast<GfxOpenClSourceScalarArg>(
      static_cast<uint32_t>(GfxOpenClSourceScalarArg::Input0Dim0) + axis);
}

GfxOpenClSourceScalarArg output0_dim_scalar_arg(size_t axis) {
  return static_cast<GfxOpenClSourceScalarArg>(
      static_cast<uint32_t>(GfxOpenClSourceScalarArg::Output0Dim0) + axis);
}

std::optional<std::vector<uint32_t>>
tile_static_u32_scalars(const std::shared_ptr<const ov::Node> &node) {
  auto tile = ov::as_type_ptr<const ov::op::v0::Tile>(node);
  if (!tile || tile->get_input_size() != 2 || tile->get_output_size() != 1 ||
      !tile->get_input_partial_shape(0).is_static() ||
      !tile->get_output_partial_shape(0).is_static() ||
      tile->get_input_element_type(0) != tile->get_output_element_type(0) ||
      (!is_f32_tile_type(tile->get_input_element_type(0)) &&
       !is_f16_tile_type(tile->get_input_element_type(0)))) {
    return std::nullopt;
  }
  const auto &input_shape = tile->get_input_shape(0);
  const auto &output_shape = tile->get_output_shape(0);
  const size_t rank = input_shape.size();
  if (rank == 0 || rank > 4 || output_shape.size() != rank ||
      ov::shape_size(input_shape) == 0 || ov::shape_size(output_shape) == 0) {
    return std::nullopt;
  }
  const auto repeats = constant_i64_vector_input(node, 1);
  if (!repeats || repeats->size() != rank) {
    return std::nullopt;
  }
  for (size_t axis = 0; axis < rank; ++axis) {
    if (input_shape[axis] == 0 || (*repeats)[axis] <= 0 ||
        static_cast<uint64_t>(input_shape[axis]) *
                static_cast<uint64_t>((*repeats)[axis]) !=
            output_shape[axis]) {
      return std::nullopt;
    }
  }
  uint32_t total = 0;
  if (!checked_u32(ov::shape_size(output_shape), total)) {
    return std::nullopt;
  }
  (void)total;

  std::vector<uint32_t> scalars;
  scalars.reserve(17);
  scalars.push_back(static_cast<uint32_t>(rank));
  if (!append_shape_u32(output_shape, 4, scalars) ||
      !append_shape_u32(input_shape, 4, scalars) ||
      !append_strides_u32(output_shape, 4, scalars) ||
      !append_strides_u32(input_shape, 4, scalars)) {
    return std::nullopt;
  }
  return scalars;
}

std::optional<uint32_t>
tile_dynamic_static_rank(const std::shared_ptr<const ov::Node> &node) {
  if (!node || node->get_input_size() != 2 || node->get_output_size() != 1 ||
      node->get_input_element_type(0) != node->get_output_element_type(0) ||
      (!is_f32_tile_type(node->get_input_element_type(0)) &&
       !is_f16_tile_type(node->get_input_element_type(0)))) {
    return std::nullopt;
  }
  const auto input_rank = node->get_input_partial_shape(0).rank();
  if (!input_rank.is_static() || input_rank.get_length() <= 0 ||
      input_rank.get_length() > 4) {
    return std::nullopt;
  }
  const auto output_rank = node->get_output_partial_shape(0).rank();
  if (output_rank.is_static() &&
      output_rank.get_length() != input_rank.get_length()) {
    return std::nullopt;
  }
  const auto repeats_shape = node->get_input_partial_shape(1);
  if (repeats_shape.is_static()) {
    const auto repeats_count = ov::shape_size(repeats_shape.to_shape());
    if (repeats_count != static_cast<size_t>(input_rank.get_length())) {
      return std::nullopt;
    }
  }
  if (node->get_output_partial_shape(0).is_static() &&
      ov::shape_size(node->get_output_shape(0)) == 0) {
    return std::nullopt;
  }
  return static_cast<uint32_t>(input_rank.get_length());
}

std::vector<GfxOpenClSourceScalarArg> tile_dynamic_shape_scalar_args() {
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount,
      GfxOpenClSourceScalarArg::StaticU32};
  scalar_args.reserve(10);
  for (uint32_t axis = 0; axis < 4; ++axis) {
    scalar_args.push_back(output0_dim_scalar_arg(axis));
  }
  for (uint32_t axis = 0; axis < 4; ++axis) {
    scalar_args.push_back(input0_dim_scalar_arg(axis));
  }
  return scalar_args;
}

GfxKernelStageManifest make_opencl_tile_manifest(std::string specialization_key,
                                                 std::string entry_point,
                                                 uint32_t scalar_arg_count) {
  GfxKernelExternalBufferAbiSpec abi{};
  abi.valid = true;
  abi.roles.push_back(GfxKernelBufferRole::TensorInput);
  abi.roles.push_back(GfxKernelBufferRole::TensorOutput);
  abi.roles.insert(abi.roles.end(), scalar_arg_count,
                   GfxKernelBufferRole::ScalarParam);

  constexpr GfxKernelFamily kFamily = GfxKernelFamily::TransposePackND;
  auto custom = make_gfx_custom_kernel_manifest(
      gfx_kernel_family_name(kFamily), gfx_kernel_family_abi_id(kFamily),
      std::move(entry_point), std::move(abi),
      make_gfx_kernel_linear_dispatch_policy(
          /*threads_per_threadgroup=*/64,
          /*precompiled_binary_required=*/false));
  return make_gfx_custom_kernel_stage_manifest(
      GfxKernelStageFamily::Layout, GfxKernelBackendDomain::OpenCl,
      GfxKernelStorageKind::Buffer, std::move(specialization_key),
      std::move(custom));
}

GfxOpenClSourceArtifact
make_tile_artifact(const GfxKernelSource &source,
                   std::string specialization_key,
                   std::vector<GfxOpenClSourceScalarArg> scalar_args,
                   std::vector<uint32_t> static_u32_scalars) {
  GfxOpenClSourceArtifact artifact{};
  artifact.stage_manifest = make_opencl_tile_manifest(
      std::move(specialization_key), source.entry_point,
      static_cast<uint32_t>(scalar_args.size()));
  artifact.valid = artifact.stage_manifest.valid;
  artifact.artifact_ref = make_gfx_kernel_artifact_ref(artifact.stage_manifest);
  artifact.artifact_ref.source_id = source.kernel_id;
  artifact.artifact_ref.entry_point = source.entry_point;
  artifact.source = source.source;
  artifact.scalar_args = std::move(scalar_args);
  artifact.static_u32_scalars = std::move(static_u32_scalars);
  artifact.direct_input_indices = {0};
  artifact.element_count_source = GfxOpenClSourceElementCountSource::Output0;
  artifact.op = GfxOpenClArtifactOp::Identity;
  artifact.input_mode = GfxOpenClArtifactInputMode::Direct;
  artifact.arg_count = static_cast<uint32_t>(
      materialize_gfx_kernel_external_buffer_roles(
          artifact.stage_manifest.custom_kernel.external_buffer_abi)
          .size());
  artifact.direct_input_count = 1;
  artifact.direct_output_count = 1;
  artifact.valid =
      artifact.valid && artifact.artifact_ref.valid &&
      artifact.artifact_ref.kind == GfxKernelArtifactKind::OpenClSource &&
      !artifact.source.empty();
  return artifact;
}

bool requested_unit_matches(std::string_view requested,
                            std::string_view actual) noexcept {
  return requested.empty() || requested == actual;
}

const GfxKernelSource *tile_kernel_source(const ov::element::Type &type,
                                          bool dynamic_shape) noexcept {
  if (is_f16_tile_type(type)) {
    return dynamic_shape ? &opencl_generated_tile_dynamic_f16_kernel_source()
                         : &opencl_generated_tile_f16_kernel_source();
  }
  if (is_f32_tile_type(type)) {
    return dynamic_shape ? &opencl_generated_tile_dynamic_f32_kernel_source()
                         : &opencl_generated_tile_f32_kernel_source();
  }
  return nullptr;
}

std::string tile_type_suffix(const ov::element::Type &type) {
  return is_f16_tile_type(type) ? "f16" : "f32";
}

} // namespace

std::optional<GfxOpenClSourceArtifact>
make_opencl_tile_source_artifact(const std::shared_ptr<const ov::Node> &node,
                                 std::string_view requested_kernel_unit_id) {
  const auto tile = ov::as_type_ptr<const ov::op::v0::Tile>(node);
  if (!tile) {
    return std::nullopt;
  }

  auto static_u32_scalars = tile_static_u32_scalars(node);
  const bool dynamic_shape = !static_u32_scalars.has_value();
  const auto *source =
      tile_kernel_source(tile->get_output_element_type(0), dynamic_shape);
  if (!source ||
      !requested_unit_matches(requested_kernel_unit_id, source->kernel_id)) {
    return std::nullopt;
  }

  if (dynamic_shape) {
    const auto rank = tile_dynamic_static_rank(node);
    if (!rank) {
      return std::nullopt;
    }
    return make_tile_artifact(
        *source,
        "opencl:generated:Tile:" +
            tile_type_suffix(tile->get_output_element_type(0)) +
            ":dynamic_static_rank",
        tile_dynamic_shape_scalar_args(), {*rank});
  }

  if (!static_u32_scalars) {
    return std::nullopt;
  }
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), static_u32_scalars->size(),
                     GfxOpenClSourceScalarArg::StaticU32);
  return make_tile_artifact(
      *source,
      "opencl:generated:Tile:" +
          tile_type_suffix(tile->get_output_element_type(0)),
      std::move(scalar_args), std::move(*static_u32_scalars));
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
