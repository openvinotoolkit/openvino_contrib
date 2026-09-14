// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/compiler/opencl_softmax_kernel_unit.hpp"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kernel_ir/gfx_custom_kernel_families.hpp"
#include "kernel_ir/opencl_kernels/softmax_kernel.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/softmax.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

bool is_f16_softmax_type(const ov::element::Type &type) {
  return type == ov::element::f16;
}

bool is_f32_softmax_type(const ov::element::Type &type) {
  return type == ov::element::f32;
}

const char *softmax_type_suffix(const ov::element::Type &type) {
  if (is_f16_softmax_type(type)) {
    return "f16";
  }
  return "f32";
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
  for (size_t i = begin; i < end; ++i) {
    product *= shape[i];
  }
  return product;
}

std::optional<size_t> normalize_axis(int64_t axis, size_t rank) {
  if (rank == 0) {
    return std::nullopt;
  }
  int64_t normalized = axis;
  if (normalized < 0) {
    normalized += static_cast<int64_t>(rank);
  }
  if (normalized < 0 || normalized >= static_cast<int64_t>(rank)) {
    return std::nullopt;
  }
  return static_cast<size_t>(normalized);
}

std::optional<int64_t>
softmax_axis(const std::shared_ptr<const ov::Node> &node) {
  if (auto softmax_v8 = ov::as_type_ptr<const ov::op::v8::Softmax>(node)) {
    return softmax_v8->get_axis();
  }
  if (auto softmax_v1 = ov::as_type_ptr<const ov::op::v1::Softmax>(node)) {
    return static_cast<int64_t>(softmax_v1->get_axis());
  }
  return std::nullopt;
}

std::optional<std::vector<uint32_t>>
softmax_static_u32_scalars(const std::shared_ptr<const ov::Node> &node) {
  const auto raw_axis = softmax_axis(node);
  if (!raw_axis) {
    return std::nullopt;
  }

  const auto element_type = node->get_input_element_type(0);
  if (node->get_input_size() != 1 || node->get_output_size() != 1 ||
      !node->get_input_partial_shape(0).is_static() ||
      !node->get_output_partial_shape(0).is_static() ||
      element_type != node->get_output_element_type(0) ||
      (!is_f32_softmax_type(element_type) &&
       !is_f16_softmax_type(element_type))) {
    return std::nullopt;
  }

  const auto &input_shape = node->get_input_shape(0);
  const auto &output_shape = node->get_output_shape(0);
  const size_t rank = input_shape.size();
  if (rank == 0 || output_shape != input_shape ||
      ov::shape_size(output_shape) == 0) {
    return std::nullopt;
  }
  const auto axis = normalize_axis(*raw_axis, rank);
  if (!axis || input_shape[*axis] == 0) {
    return std::nullopt;
  }

  uint32_t outer = 0;
  uint32_t axis_dim = 0;
  uint32_t inner = 0;
  if (!checked_u32(shape_product_range(input_shape, 0, *axis), outer) ||
      !checked_u32(input_shape[*axis], axis_dim) ||
      !checked_u32(shape_product_range(input_shape, *axis + 1, rank), inner)) {
    return std::nullopt;
  }

  return std::vector<uint32_t>{outer, axis_dim, inner};
}

std::optional<std::vector<uint32_t>> softmax_dynamic_static_rank_scalars(
    const std::shared_ptr<const ov::Node> &node) {
  const auto raw_axis = softmax_axis(node);
  if (!raw_axis) {
    return std::nullopt;
  }

  const auto element_type = node->get_input_element_type(0);
  if (node->get_input_size() != 1 || node->get_output_size() != 1 ||
      element_type != node->get_output_element_type(0) ||
      (!is_f32_softmax_type(element_type) &&
       !is_f16_softmax_type(element_type))) {
    return std::nullopt;
  }
  const auto input_rank = node->get_input_partial_shape(0).rank();
  const auto output_rank = node->get_output_partial_shape(0).rank();
  if (!input_rank.is_static() || !output_rank.is_static() ||
      input_rank.get_length() != output_rank.get_length() ||
      input_rank.get_length() <= 0 || input_rank.get_length() > 8) {
    return std::nullopt;
  }
  if (node->get_output_partial_shape(0).is_static()) {
    return std::nullopt;
  }

  const auto rank = static_cast<size_t>(input_rank.get_length());
  const auto axis = normalize_axis(*raw_axis, rank);
  if (!axis) {
    return std::nullopt;
  }

  return std::vector<uint32_t>{static_cast<uint32_t>(rank),
                               static_cast<uint32_t>(*axis)};
}

std::vector<GfxOpenClSourceScalarArg> softmax_dynamic_shape_scalar_args() {
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount,
      GfxOpenClSourceScalarArg::StaticU32, GfxOpenClSourceScalarArg::StaticU32};
  scalar_args.reserve(11);
  for (uint32_t axis = 0; axis < 8; ++axis) {
    scalar_args.push_back(static_cast<GfxOpenClSourceScalarArg>(
        static_cast<uint32_t>(GfxOpenClSourceScalarArg::Input0Dim0) + axis));
  }
  return scalar_args;
}

const GfxKernelSource *softmax_kernel_source(const ov::element::Type &type,
                                             bool dynamic_shape) {
  if (is_f32_softmax_type(type)) {
    return dynamic_shape ? &opencl_generated_softmax_f32_dynamic_kernel_source()
                         : &opencl_generated_softmax_f32_kernel_source();
  }
  if (is_f16_softmax_type(type)) {
    return dynamic_shape ? &opencl_generated_softmax_f16_dynamic_kernel_source()
                         : &opencl_generated_softmax_f16_kernel_source();
  }
  return nullptr;
}

GfxKernelStageManifest
make_opencl_softmax_manifest(std::string specialization_key,
                             std::string entry_point,
                             uint32_t scalar_arg_count) {
  GfxKernelExternalBufferAbiSpec abi{};
  abi.valid = true;
  abi.roles.push_back(GfxKernelBufferRole::TensorInput);
  abi.roles.push_back(GfxKernelBufferRole::TensorOutput);
  abi.roles.insert(abi.roles.end(), scalar_arg_count,
                   GfxKernelBufferRole::ScalarParam);

  constexpr GfxKernelFamily kFamily = GfxKernelFamily::SoftmaxBuffer;
  auto custom = make_gfx_custom_kernel_manifest(
      gfx_kernel_family_name(kFamily), gfx_kernel_family_abi_id(kFamily),
      std::move(entry_point), std::move(abi),
      make_gfx_kernel_linear_dispatch_policy(
          /*threads_per_threadgroup=*/64,
          /*precompiled_binary_required=*/false));
  return make_gfx_custom_kernel_stage_manifest(
      GfxKernelStageFamily::Softmax, GfxKernelBackendDomain::OpenCl,
      GfxKernelStorageKind::Buffer, std::move(specialization_key),
      std::move(custom));
}

GfxOpenClSourceArtifact
make_softmax_artifact(const GfxKernelSource &source,
                      std::string specialization_key,
                      std::vector<GfxOpenClSourceScalarArg> scalar_args,
                      std::vector<uint32_t> static_u32_scalars) {
  GfxOpenClSourceArtifact artifact{};
  artifact.stage_manifest = make_opencl_softmax_manifest(
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
  artifact.op = GfxOpenClArtifactOp::Softmax;
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

} // namespace

std::optional<GfxOpenClSourceArtifact>
make_opencl_softmax_source_artifact(const std::shared_ptr<const ov::Node> &node,
                                    std::string_view requested_kernel_unit_id) {
  if (!softmax_axis(node)) {
    return std::nullopt;
  }
  if (node->get_input_size() != 1 || node->get_output_size() != 1) {
    return std::nullopt;
  }
  const auto element_type = node->get_input_element_type(0);
  if (!is_f32_softmax_type(element_type) &&
      !is_f16_softmax_type(element_type)) {
    return std::nullopt;
  }
  const std::string type_suffix = softmax_type_suffix(element_type);

  if (auto static_u32_scalars = softmax_static_u32_scalars(node)) {
    const auto *source =
        softmax_kernel_source(element_type, /*dynamic_shape=*/false);
    if (!source ||
        !requested_unit_matches(requested_kernel_unit_id, source->kernel_id)) {
      return std::nullopt;
    }
    std::vector<GfxOpenClSourceScalarArg> scalar_args = {
        GfxOpenClSourceScalarArg::ElementCount};
    scalar_args.insert(scalar_args.end(), static_u32_scalars->size(),
                       GfxOpenClSourceScalarArg::StaticU32);
    return make_softmax_artifact(
        *source, "opencl:generated:Softmax:" + type_suffix,
        std::move(scalar_args), std::move(*static_u32_scalars));
  }

  if (auto dynamic_static_rank = softmax_dynamic_static_rank_scalars(node)) {
    const auto *source =
        softmax_kernel_source(element_type, /*dynamic_shape=*/true);
    if (!source ||
        !requested_unit_matches(requested_kernel_unit_id, source->kernel_id)) {
      return std::nullopt;
    }
    return make_softmax_artifact(
        *source,
        "opencl:generated:Softmax:" + type_suffix + ":dynamic_static_rank",
        softmax_dynamic_shape_scalar_args(), std::move(*dynamic_static_rank));
  }

  return std::nullopt;
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
