// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/compiler/opencl_range_kernel_unit.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kernel_ir/gfx_custom_kernel_families.hpp"
#include "kernel_ir/opencl_kernels/range_kernel.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/range.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

bool is_f16_range_type(const ov::element::Type &type) {
  return type == ov::element::f16;
}

bool is_f32_range_type(const ov::element::Type &type) {
  return type == ov::element::f32;
}

bool is_i64_range_type(const ov::element::Type &type) {
  return type == ov::element::i64;
}

bool is_static_single_element_input(const std::shared_ptr<const ov::Node> &node,
                                    size_t input_idx) {
  return node && input_idx < node->get_input_size() &&
         node->get_input_partial_shape(input_idx).is_static() &&
         ov::shape_size(node->get_input_shape(input_idx)) == 1;
}

std::optional<int64_t> static_rank(const ov::PartialShape &shape) {
  const auto rank = shape.rank();
  if (!rank.is_static()) {
    return std::nullopt;
  }
  return rank.get_length();
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

bool range_dynamic_i64_unit_supported(
    const std::shared_ptr<const ov::Node> &node) {
  if (!node || !ov::as_type_ptr<const ov::op::v4::Range>(node) ||
      node->get_input_size() != 3 || node->get_output_size() != 1 ||
      node->get_output_partial_shape(0).is_static() ||
      node->get_output_element_type(0) != ov::element::i64 ||
      node->get_input_element_type(1) != ov::element::i64 ||
      !is_static_single_element_input(node, 1)) {
    return false;
  }
  const auto rank = static_rank(node->get_output_partial_shape(0));
  if (!rank || *rank != 1) {
    return false;
  }
  const auto start = constant_i64_vector_input(node, 0);
  const auto step = constant_i64_vector_input(node, 2);
  return start && step && start->size() == 1 && step->size() == 1 &&
         (*start)[0] == 0 && (*step)[0] == 1;
}

bool range_has_static_generated_artifact(
    const std::shared_ptr<const ov::Node> &node) {
  if (!node || !ov::as_type_ptr<const ov::op::v4::Range>(node) ||
      node->get_input_size() != 3 || node->get_output_size() != 1 ||
      !node->get_output_partial_shape(0).is_static() ||
      !is_static_single_element_input(node, 0) ||
      !is_static_single_element_input(node, 1) ||
      !is_static_single_element_input(node, 2)) {
    return false;
  }
  const auto output_type = node->get_output_element_type(0);
  if (!is_f16_range_type(output_type) && !is_f32_range_type(output_type) &&
      !is_i64_range_type(output_type)) {
    return false;
  }
  if (node->get_input_element_type(0) != output_type ||
      node->get_input_element_type(1) != output_type ||
      node->get_input_element_type(2) != output_type) {
    return false;
  }
  const auto &output_shape = node->get_output_shape(0);
  return output_shape.size() == 1 && ov::shape_size(output_shape) > 0;
}

GfxKernelStageManifest
make_opencl_range_manifest(std::string specialization_key,
                           std::string entry_point, uint32_t direct_inputs,
                           uint32_t direct_outputs, uint32_t scalar_arg_count) {
  GfxKernelExternalBufferAbiSpec abi{};
  abi.valid = true;
  abi.roles.insert(abi.roles.end(), direct_inputs,
                   GfxKernelBufferRole::TensorInput);
  abi.roles.insert(abi.roles.end(), direct_outputs,
                   GfxKernelBufferRole::TensorOutput);
  abi.roles.insert(abi.roles.end(), scalar_arg_count,
                   GfxKernelBufferRole::ScalarParam);

  constexpr GfxKernelFamily kFamily = GfxKernelFamily::GatherScatterIndexed;
  auto custom = make_gfx_custom_kernel_manifest(
      gfx_kernel_family_name(kFamily), gfx_kernel_family_abi_id(kFamily),
      std::move(entry_point), std::move(abi),
      make_gfx_kernel_linear_dispatch_policy(
          /*threads_per_threadgroup=*/64,
          /*precompiled_binary_required=*/false));
  return make_gfx_custom_kernel_stage_manifest(
      GfxKernelStageFamily::GatherScatter, GfxKernelBackendDomain::OpenCl,
      GfxKernelStorageKind::Buffer, std::move(specialization_key),
      std::move(custom));
}

GfxOpenClSourceArtifact
make_range_artifact(const GfxKernelSource &source,
                    std::string specialization_key, uint32_t direct_inputs,
                    std::vector<size_t> direct_input_indices) {
  GfxOpenClSourceArtifact artifact{};
  artifact.stage_manifest = make_opencl_range_manifest(
      std::move(specialization_key), source.entry_point, direct_inputs,
      /*direct_outputs=*/1,
      /*scalar_arg_count=*/1);
  artifact.valid = artifact.stage_manifest.valid;
  artifact.artifact_ref = make_gfx_kernel_artifact_ref(artifact.stage_manifest);
  artifact.artifact_ref.source_id = source.kernel_id;
  artifact.artifact_ref.entry_point = source.entry_point;
  artifact.source = source.source;
  artifact.scalar_args = {GfxOpenClSourceScalarArg::ElementCount};
  artifact.direct_input_indices = std::move(direct_input_indices);
  artifact.element_count_source = GfxOpenClSourceElementCountSource::Output0;
  artifact.op = GfxOpenClArtifactOp::Identity;
  artifact.input_mode = GfxOpenClArtifactInputMode::Direct;
  artifact.arg_count = static_cast<uint32_t>(
      materialize_gfx_kernel_external_buffer_roles(
          artifact.stage_manifest.custom_kernel.external_buffer_abi)
          .size());
  artifact.direct_input_count =
      static_cast<uint32_t>(artifact.direct_input_indices.size());
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

const GfxKernelSource *
static_range_kernel_source(const ov::element::Type &type) noexcept {
  if (is_f32_range_type(type)) {
    return &opencl_generated_range_f32_kernel_source();
  }
  if (is_f16_range_type(type)) {
    return &opencl_generated_range_f16_kernel_source();
  }
  if (is_i64_range_type(type)) {
    return &opencl_generated_range_i64_kernel_source();
  }
  return nullptr;
}

} // namespace

std::optional<GfxOpenClSourceArtifact>
make_opencl_range_source_artifact(const std::shared_ptr<const ov::Node> &node,
                                  std::string_view requested_kernel_unit_id) {
  if (range_dynamic_i64_unit_supported(node)) {
    const auto &source = opencl_generated_range_i64_unit_kernel_source();
    if (!requested_unit_matches(requested_kernel_unit_id, source.kernel_id)) {
      return std::nullopt;
    }
    return make_range_artifact(source,
                               "opencl:generated:Range:i64:dynamic_unit",
                               /*direct_inputs=*/1, {1});
  }

  if (!range_has_static_generated_artifact(node)) {
    return std::nullopt;
  }
  const auto *source =
      static_range_kernel_source(node->get_output_element_type(0));
  if (!source ||
      !requested_unit_matches(requested_kernel_unit_id, source->kernel_id)) {
    return std::nullopt;
  }
  std::string type_suffix = "unknown";
  if (is_f32_range_type(node->get_output_element_type(0))) {
    type_suffix = "f32";
  } else if (is_f16_range_type(node->get_output_element_type(0))) {
    type_suffix = "f16";
  } else if (is_i64_range_type(node->get_output_element_type(0))) {
    type_suffix = "i64";
  }
  return make_range_artifact(*source, "opencl:generated:Range:" + type_suffix,
                             /*direct_inputs=*/3, {0, 1, 2});
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
