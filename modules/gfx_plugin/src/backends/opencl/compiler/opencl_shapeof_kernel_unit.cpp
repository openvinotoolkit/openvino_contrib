// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/compiler/opencl_shapeof_kernel_unit.hpp"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "backends/opencl/compiler/opencl_kernel_artifacts.hpp"
#include "compiler/executable_bundle.hpp"
#include "compiler/lowering_planner.hpp"
#include "kernel_ir/gfx_custom_kernel_families.hpp"
#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "kernel_ir/opencl_kernels/shapeof_kernel.hpp"
#include "openvino/op/util/shape_of_base.hpp"

namespace ov {
namespace gfx_plugin {

namespace {

bool is_opencl_shapeof_output_type(const ov::element::Type &type) {
  return type == ov::element::i32 || type == ov::element::i64;
}

std::optional<size_t>
opencl_shapeof_rank(const std::shared_ptr<const ov::Node> &node) {
  const auto shape_of = ov::as_type_ptr<const ov::op::util::ShapeOfBase>(node);
  if (!shape_of || shape_of->get_input_size() != 1 ||
      shape_of->get_output_size() != 1 ||
      !shape_of->get_input_partial_shape(0).rank().is_static() ||
      !shape_of->get_output_partial_shape(0).is_static()) {
    return std::nullopt;
  }

  const auto output_type = shape_of->get_output_element_type(0);
  if (!is_opencl_shapeof_output_type(output_type)) {
    return std::nullopt;
  }

  const auto &output_shape = shape_of->get_output_shape(0);
  const size_t rank = static_cast<size_t>(
      shape_of->get_input_partial_shape(0).rank().get_length());
  if (rank == 0 || rank > 8 || output_shape.size() != 1 ||
      output_shape[0] != rank) {
    return std::nullopt;
  }
  return rank;
}

GfxKernelStageManifest
make_opencl_shapeof_manifest(std::string specialization_key,
                             std::string entry_point,
                             uint32_t scalar_arg_count) {
  GfxKernelExternalBufferAbiSpec abi{};
  abi.valid = true;
  abi.roles.push_back(GfxKernelBufferRole::TensorInput);
  abi.roles.push_back(GfxKernelBufferRole::TensorOutput);
  abi.roles.insert(abi.roles.end(), scalar_arg_count,
                   GfxKernelBufferRole::ScalarParam);

  const auto kernel_family = GfxKernelFamily::GatherScatterIndexed;
  auto custom = make_gfx_custom_kernel_manifest(
      gfx_kernel_family_name(kernel_family),
      gfx_kernel_family_abi_id(kernel_family), std::move(entry_point),
      std::move(abi),
      make_gfx_kernel_linear_dispatch_policy(
          /*threads_per_threadgroup=*/64,
          /*precompiled_binary_required=*/false));

  return make_gfx_custom_kernel_stage_manifest(
      GfxKernelStageFamily::GatherScatter, GfxKernelBackendDomain::OpenCl,
      GfxKernelStorageKind::Buffer, std::move(specialization_key),
      std::move(custom));
}

std::vector<GfxOpenClSourceScalarArg> make_shapeof_scalar_args() {
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.reserve(9);
  for (uint32_t axis = 0; axis < 8; ++axis) {
    scalar_args.push_back(static_cast<GfxOpenClSourceScalarArg>(
        static_cast<uint32_t>(GfxOpenClSourceScalarArg::Input0Dim0) + axis));
  }
  return scalar_args;
}

GfxOpenClSourceArtifact
make_shapeof_artifact(GfxKernelStageManifest manifest, std::string source_id,
                      std::vector<GfxOpenClSourceScalarArg> scalar_args,
                      std::string source) {
  GfxOpenClSourceArtifact artifact{};
  artifact.valid = manifest.valid;
  artifact.stage_manifest = std::move(manifest);
  artifact.artifact_ref = make_gfx_kernel_artifact_ref(artifact.stage_manifest);
  artifact.artifact_ref.source_id = std::move(source_id);
  artifact.artifact_ref.entry_point =
      artifact.stage_manifest.custom_kernel.entry_point;
  artifact.source = std::move(source);
  artifact.scalar_args = std::move(scalar_args);
  artifact.direct_input_indices = {0};

  const auto roles = materialize_gfx_kernel_external_buffer_roles(
      artifact.stage_manifest.custom_kernel.external_buffer_abi);
  artifact.arg_count = static_cast<uint32_t>(roles.size());
  artifact.direct_input_count =
      static_cast<uint32_t>(artifact.direct_input_indices.size());
  artifact.direct_output_count = static_cast<uint32_t>(std::count(
      roles.begin(), roles.end(), GfxKernelBufferRole::TensorOutput));
  artifact.op = GfxOpenClArtifactOp::Identity;
  artifact.input_mode = GfxOpenClArtifactInputMode::Direct;
  artifact.valid =
      artifact.valid && artifact.artifact_ref.valid &&
      artifact.artifact_ref.kind == GfxKernelArtifactKind::OpenClSource &&
      !artifact.source.empty();
  return artifact;
}

} // namespace

std::optional<GfxOpenClSourceArtifact>
make_opencl_shapeof_source_artifact(const std::shared_ptr<const ov::Node> &node,
                                    std::string_view expected_source_id) {
  const auto rank = opencl_shapeof_rank(node);
  if (!rank) {
    return std::nullopt;
  }

  const bool output_i64 = node->get_output_element_type(0) == ov::element::i64;
  const std::string source_id = output_i64 ? "opencl/generated/shapeof_i64"
                                           : "opencl/generated/shapeof_i32";
  if (!expected_source_id.empty() && source_id != expected_source_id) {
    return std::nullopt;
  }

  const std::string entry_point = output_i64
                                      ? "gfx_opencl_generated_shapeof_i64"
                                      : "gfx_opencl_generated_shapeof_i32";
  auto manifest = make_opencl_shapeof_manifest(
      "opencl:generated:shapeof:" + std::string(output_i64 ? "i64" : "i32") +
          ":rank" + std::to_string(*rank),
      entry_point,
      /*scalar_arg_count=*/9);
  return make_shapeof_artifact(
      std::move(manifest), source_id, make_shapeof_scalar_args(),
      output_i64 ? opencl_generated_shapeof_i64_kernel_source().source
                 : opencl_generated_shapeof_i32_kernel_source().source);
}

namespace compiler {

bool is_opencl_shapeof_node(const std::shared_ptr<const ov::Node> &node) {
  return static_cast<bool>(
      ov::as_type_ptr<const ov::op::util::ShapeOfBase>(node));
}

namespace {

std::shared_ptr<const ::ov::gfx_plugin::KernelArtifactPayload>
materialize_shapeof_payload(const KernelArtifactDescriptor &descriptor,
                            GfxOpenClSourceArtifact artifact) {
  if (!opencl_source_artifact_matches_descriptor_contract(descriptor,
                                                          artifact)) {
    return {};
  }
  return std::make_shared<GfxOpenClSourceArtifactPayload>(std::move(artifact));
}

} // namespace

KernelUnit
resolve_opencl_shapeof_kernel_unit(const std::shared_ptr<const ov::Node> &node,
                                   const KernelRegistry &registry) {
  const auto artifact = make_opencl_shapeof_source_artifact(node);
  if (!artifact || !artifact->valid) {
    return {};
  }
  return registry.resolve(LoweringRouteKind::GeneratedKernel,
                          artifact->artifact_ref.source_id);
}

OperationSupportResult
query_opencl_shapeof_operation(const std::shared_ptr<const ov::Node> &node,
                               const KernelRegistry &registry) {
  if (!is_opencl_shapeof_node(node)) {
    return make_unsupported_operation("not_opencl_shapeof");
  }
  const auto unit = resolve_opencl_shapeof_kernel_unit(node, registry);
  if (!unit.valid()) {
    return make_unsupported_operation("missing_opencl_shapeof_kernel_unit");
  }
  return make_supported_operation("registered_opencl_shapeof_kernel_unit",
                                  unit.route_kind(), 0.55, unit.id());
}

std::shared_ptr<const ::ov::gfx_plugin::KernelArtifactPayload>
build_opencl_shapeof_kernel_artifact_payload(
    const KernelArtifactDescriptor &descriptor, const PlannedOperation &op) {
  if (descriptor.kernel.backend_domain != "opencl" ||
      descriptor.payload_kind != KernelArtifactPayloadKind::OpenClSource ||
      !op.source_node || !is_opencl_shapeof_node(op.source_node)) {
    return {};
  }

  auto artifact = make_opencl_shapeof_source_artifact(
      op.source_node, descriptor.kernel.kernel_id);
  if (!artifact || !artifact->valid) {
    return {};
  }
  return materialize_shapeof_payload(descriptor, std::move(*artifact));
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
