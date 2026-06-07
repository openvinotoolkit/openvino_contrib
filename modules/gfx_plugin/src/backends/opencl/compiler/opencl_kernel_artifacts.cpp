// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/compiler/opencl_kernel_artifacts.hpp"

#include <algorithm>
#include <memory>
#include <string_view>
#include <utility>

#include "backends/opencl/compiler/opencl_conv_kernel_unit.hpp"
#include "backends/opencl/compiler/opencl_pool_kernel_unit.hpp"
#include "backends/opencl/compiler/opencl_range_kernel_unit.hpp"
#include "backends/opencl/compiler/opencl_softmax_kernel_unit.hpp"
#include "backends/opencl/compiler/opencl_tile_kernel_unit.hpp"
#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "kernel_ir/opencl_kernels/conv2d_kernel.hpp"
#include "kernel_ir/opencl_kernels/pool2d_kernel.hpp"
#include "kernel_ir/opencl_kernels/range_kernel.hpp"
#include "kernel_ir/opencl_kernels/softmax_kernel.hpp"
#include "kernel_ir/opencl_kernels/tile_kernel.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

bool starts_with(std::string_view value, std::string_view prefix) noexcept {
  return value.size() >= prefix.size() &&
         value.substr(0, prefix.size()) == prefix;
}

bool source_artifact_matches_descriptor(
    const KernelArtifactDescriptor &descriptor,
    const GfxOpenClSourceArtifact &artifact) noexcept {
  return artifact.valid && artifact.artifact_ref.valid &&
         artifact.artifact_ref.kind == GfxKernelArtifactKind::OpenClSource &&
         artifact.artifact_ref.backend_domain ==
             GfxKernelBackendDomain::OpenCl &&
         descriptor.kernel.backend_domain == "opencl" &&
         descriptor.payload_kind == KernelArtifactPayloadKind::OpenClSource &&
         descriptor.kernel.kernel_id == artifact.artifact_ref.source_id &&
         descriptor.kernel.origin == classify_opencl_kernel_artifact_origin(
                                         artifact.artifact_ref.source_id);
}

uint32_t count_runtime_param_roles(const GfxKernelStageManifest &manifest) {
  if (!manifest.valid || !manifest.custom_kernel.valid ||
      !manifest.custom_kernel.external_buffer_abi.valid) {
    return 0;
  }
  const auto roles = materialize_gfx_kernel_external_buffer_roles(
      manifest.custom_kernel.external_buffer_abi);
  return static_cast<uint32_t>(std::count(roles.begin(), roles.end(),
                                          GfxKernelBufferRole::RuntimeParams));
}

KernelLaunchPlanDescriptor
make_opencl_launch_plan_descriptor(const GfxOpenClSourceArtifact &artifact) {
  KernelLaunchPlanDescriptor descriptor;
  if (!artifact.stage_manifest.valid ||
      !artifact.stage_manifest.custom_kernel.valid ||
      !artifact.stage_manifest.custom_kernel.external_buffer_abi.valid) {
    return descriptor;
  }
  const auto roles = materialize_gfx_kernel_external_buffer_roles(
      artifact.stage_manifest.custom_kernel.external_buffer_abi);
  if (roles.empty()) {
    return descriptor;
  }
  descriptor.valid = true;
  descriptor.buffer_roles.reserve(roles.size());
  for (const auto role : roles) {
    descriptor.buffer_roles.emplace_back(
        kernel_buffer_role_descriptor_name(role));
  }
  descriptor.direct_input_indices = artifact.direct_input_indices;
  descriptor.input_arg_count = artifact.direct_input_count;
  descriptor.scalar_arg_kinds.reserve(artifact.scalar_args.size());
  for (const auto scalar : artifact.scalar_args) {
    descriptor.scalar_arg_kinds.push_back(static_cast<uint32_t>(scalar));
  }
  return descriptor;
}

std::shared_ptr<const KernelArtifactPayload>
materialize_descriptor_owned_opencl_payload(
    const KernelArtifactDescriptor &descriptor,
    GfxOpenClSourceArtifact artifact) {
  if (!is_explicit_opencl_source_artifact_unit(
          artifact.artifact_ref.source_id)) {
    return {};
  }
  if (!opencl_source_artifact_matches_descriptor_contract(descriptor,
                                                          artifact)) {
    return {};
  }
  return std::make_shared<GfxOpenClSourceArtifactPayload>(std::move(artifact));
}

std::shared_ptr<const KernelArtifactPayload>
resolve_opencl_payload(const KernelArtifactDescriptor &descriptor,
                       const PlannedOperation &op) {
  if (descriptor.kernel.backend_domain != "opencl" ||
      descriptor.payload_kind != KernelArtifactPayloadKind::OpenClSource ||
      !op.source_node) {
    return {};
  }

  if (auto conv2d_payload =
          build_opencl_conv2d_kernel_artifact_payload(descriptor, op)) {
    return conv2d_payload;
  }
  if (is_opencl_conv2d_node(op.source_node)) {
    return {};
  }
  if (auto range_payload =
          build_opencl_range_kernel_artifact_payload(descriptor, op)) {
    return range_payload;
  }
  if (is_opencl_range_node(op.source_node)) {
    return {};
  }
  if (auto tile_payload =
          build_opencl_tile_kernel_artifact_payload(descriptor, op)) {
    return tile_payload;
  }
  if (is_opencl_tile_node(op.source_node)) {
    return {};
  }
  if (auto softmax_payload =
          build_opencl_softmax_kernel_artifact_payload(descriptor, op)) {
    return softmax_payload;
  }
  if (is_opencl_softmax_node(op.source_node)) {
    return {};
  }
  if (auto pool2d_payload =
          build_opencl_pool2d_kernel_artifact_payload(descriptor, op)) {
    return pool2d_payload;
  }
  if (is_opencl_pool2d_node(op.source_node)) {
    return {};
  }

  auto source_artifact = resolve_gfx_opencl_source_artifact(op.source_node);
  if (!source_artifact || !source_artifact->valid ||
      !is_explicit_opencl_source_artifact_unit(
          source_artifact->artifact_ref.source_id)) {
    return {};
  }
  return materialize_descriptor_owned_opencl_payload(
      descriptor, std::move(*source_artifact));
}

} // namespace

::ov::gfx_plugin::KernelArtifactOrigin classify_opencl_kernel_artifact_origin(
    std::string_view kernel_unit_id) noexcept {
  if (starts_with(kernel_unit_id, "opencl/generated/")) {
    return KernelArtifactOrigin::Generated;
  }
  if (starts_with(kernel_unit_id, "opencl/")) {
    return KernelArtifactOrigin::HandwrittenException;
  }
  return KernelArtifactOrigin::Unknown;
}

bool is_explicit_opencl_source_artifact_unit(
    std::string_view kernel_unit_id) noexcept {
  return starts_with(kernel_unit_id, "opencl/generated/interpolate_") ||
         starts_with(kernel_unit_id, "opencl/generated/matmul_") ||
         starts_with(kernel_unit_id, "opencl/generated/shapeof_") ||
         starts_with(kernel_unit_id, "opencl/generated/concat") ||
         starts_with(kernel_unit_id, "opencl/generated/split") ||
         starts_with(kernel_unit_id, "opencl/generated/activation_") ||
         starts_with(kernel_unit_id, "opencl/generated/eltwise_") ||
         starts_with(kernel_unit_id, "opencl/generated/reduction_") ||
         starts_with(kernel_unit_id, "opencl/generated/transpose_");
}

KernelArtifactPayloadResolver make_opencl_kernel_artifact_payload_resolver() {
  return [](const KernelArtifactDescriptor &descriptor,
            const PlannedOperation &op) {
    return resolve_opencl_payload(descriptor, op);
  };
}

bool finalize_opencl_kernel_artifact_descriptor_contract(
    KernelArtifactDescriptor &descriptor,
    const ::ov::gfx_plugin::GfxOpenClSourceArtifact &artifact) {
  if (!source_artifact_matches_descriptor(descriptor, artifact)) {
    return false;
  }
  descriptor.runtime_param_buffer_count =
      count_runtime_param_roles(artifact.stage_manifest);
  descriptor.runtime_param_i64_metadata.clear();
  descriptor.runtime_param_reduce_keep_dims = false;
  descriptor.runtime_param_reduce_keep_dims_valid = false;
  descriptor.entry_point = artifact.artifact_ref.entry_point;
  descriptor.compile_options_key =
      gfx_opencl_source_artifact_build_options(artifact);
  descriptor.abi_arg_count = artifact.arg_count;
  descriptor.abi_output_arg_count = artifact.direct_output_count;
  descriptor.launch_plan = make_opencl_launch_plan_descriptor(artifact);
  finalize_kernel_artifact_descriptor_identity(descriptor);
  return true;
}

bool opencl_source_artifact_matches_descriptor_contract(
    const KernelArtifactDescriptor &descriptor,
    const ::ov::gfx_plugin::GfxOpenClSourceArtifact &artifact) {
  KernelArtifactDescriptor expected = descriptor;
  if (!finalize_opencl_kernel_artifact_descriptor_contract(expected,
                                                           artifact)) {
    return false;
  }
  return descriptor.entry_point == expected.entry_point &&
         descriptor.compile_options_key == expected.compile_options_key &&
         descriptor.abi_arg_count == expected.abi_arg_count &&
         descriptor.abi_output_arg_count == expected.abi_output_arg_count &&
         descriptor.runtime_param_buffer_count ==
             expected.runtime_param_buffer_count &&
         descriptor.runtime_param_i64_metadata ==
             expected.runtime_param_i64_metadata &&
         descriptor.runtime_param_reduce_keep_dims ==
             expected.runtime_param_reduce_keep_dims &&
         descriptor.runtime_param_reduce_keep_dims_valid ==
             expected.runtime_param_reduce_keep_dims_valid &&
         descriptor.launch_plan.valid == expected.launch_plan.valid &&
         descriptor.launch_plan.buffer_roles ==
             expected.launch_plan.buffer_roles &&
         descriptor.launch_plan.direct_input_indices ==
             expected.launch_plan.direct_input_indices &&
         descriptor.launch_plan.input_indices ==
             expected.launch_plan.input_indices &&
         descriptor.launch_plan.input_arg_count ==
             expected.launch_plan.input_arg_count &&
         descriptor.launch_plan.operand_kinds ==
             expected.launch_plan.operand_kinds &&
         descriptor.launch_plan.operand_arg_indices ==
             expected.launch_plan.operand_arg_indices &&
         descriptor.launch_plan.scalar_args ==
             expected.launch_plan.scalar_args &&
         descriptor.launch_plan.scalar_arg_kinds ==
             expected.launch_plan.scalar_arg_kinds &&
         descriptor.abi_fingerprint == expected.abi_fingerprint &&
         descriptor.manifest_ref == expected.manifest_ref &&
         descriptor.artifact_key == expected.artifact_key;
}

KernelArtifactDescriptorResolver
make_opencl_kernel_artifact_descriptor_resolver() {
  return [](KernelArtifactDescriptor &descriptor,
            const PlannedOperation &op) -> bool {
    if (descriptor.kernel.backend_domain != "opencl" ||
        descriptor.payload_kind != KernelArtifactPayloadKind::OpenClSource ||
        !op.source_node) {
      finalize_kernel_artifact_descriptor_identity(descriptor);
      return true;
    }

    if (auto artifact = make_opencl_conv2d_source_artifact(
            op.source_node, descriptor.kernel.kernel_id)) {
      return artifact->valid &&
             finalize_opencl_kernel_artifact_descriptor_contract(descriptor,
                                                                 *artifact);
    }
    if (is_opencl_conv2d_node(op.source_node)) {
      return false;
    }
    if (auto artifact = make_opencl_range_source_artifact(
            op.source_node, descriptor.kernel.kernel_id)) {
      return artifact->valid &&
             finalize_opencl_kernel_artifact_descriptor_contract(descriptor,
                                                                 *artifact);
    }
    if (is_opencl_range_node(op.source_node)) {
      return false;
    }
    if (auto artifact = make_opencl_tile_source_artifact(
            op.source_node, descriptor.kernel.kernel_id)) {
      return artifact->valid &&
             finalize_opencl_kernel_artifact_descriptor_contract(descriptor,
                                                                 *artifact);
    }
    if (is_opencl_tile_node(op.source_node)) {
      return false;
    }
    if (auto artifact = make_opencl_softmax_source_artifact(
            op.source_node, descriptor.kernel.kernel_id)) {
      return artifact->valid &&
             finalize_opencl_kernel_artifact_descriptor_contract(descriptor,
                                                                 *artifact);
    }
    if (is_opencl_softmax_node(op.source_node)) {
      return false;
    }
    if (auto artifact = make_opencl_pool2d_source_artifact(
            op.source_node, descriptor.kernel.kernel_id)) {
      return artifact->valid &&
             finalize_opencl_kernel_artifact_descriptor_contract(descriptor,
                                                                 *artifact);
    }
    if (is_opencl_pool2d_node(op.source_node)) {
      return false;
    }

    auto source_artifact = resolve_gfx_opencl_source_artifact(op.source_node);
    if (!source_artifact || !source_artifact->valid ||
        !is_explicit_opencl_source_artifact_unit(
            source_artifact->artifact_ref.source_id)) {
      return false;
    }
    return finalize_opencl_kernel_artifact_descriptor_contract(
        descriptor, *source_artifact);
  };
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
