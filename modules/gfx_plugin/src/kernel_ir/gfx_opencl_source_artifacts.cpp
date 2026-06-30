// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/gfx_opencl_source_artifacts.hpp"

#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kernel_ir/gfx_custom_kernel_families.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

GfxKernelFamily opencl_kernel_family_for_stage(GfxKernelStageFamily family) {
  switch (family) {
  case GfxKernelStageFamily::Reduction:
    return GfxKernelFamily::ReductionBuffer;
  case GfxKernelStageFamily::Softmax:
    return GfxKernelFamily::SoftmaxBuffer;
  case GfxKernelStageFamily::Pooling:
    return GfxKernelFamily::Pool2DWindow;
  case GfxKernelStageFamily::Gemm:
    return GfxKernelFamily::MatMulBuffer;
  case GfxKernelStageFamily::Resize:
  case GfxKernelStageFamily::Layout:
  case GfxKernelStageFamily::Transpose:
    return GfxKernelFamily::TransposePackND;
  case GfxKernelStageFamily::ConcatSplit:
    return GfxKernelFamily::ConcatSplitGeneric;
  case GfxKernelStageFamily::GatherScatter:
    return GfxKernelFamily::GatherScatterIndexed;
  case GfxKernelStageFamily::Convolution:
  case GfxKernelStageFamily::GroupConvolution:
    return GfxKernelFamily::Conv2DDirect;
  case GfxKernelStageFamily::Activation:
  case GfxKernelStageFamily::Eltwise:
  default:
    return GfxKernelFamily::EltwiseFusedBuffer;
  }
}

} // namespace

GfxKernelStageManifest make_opencl_source_manifest(
    GfxKernelStageFamily family, std::string specialization_key,
    std::string entry_point, uint32_t direct_inputs, uint32_t scalar_arg_count,
    uint32_t direct_outputs) {
  GfxKernelExternalBufferAbiSpec abi{};
  abi.valid = true;
  abi.roles.insert(abi.roles.end(), direct_inputs,
                   GfxKernelBufferRole::TensorInput);
  abi.roles.insert(abi.roles.end(), direct_outputs,
                   GfxKernelBufferRole::TensorOutput);
  abi.roles.insert(abi.roles.end(), scalar_arg_count,
                   GfxKernelBufferRole::ScalarParam);

  const auto kernel_family = opencl_kernel_family_for_stage(family);
  auto dispatch = make_gfx_kernel_linear_dispatch_policy(
      /*threads_per_threadgroup=*/64,
      /*precompiled_binary_required=*/false);
  auto custom = make_gfx_custom_kernel_manifest(
      gfx_kernel_family_name(kernel_family),
      gfx_kernel_family_abi_id(kernel_family), std::move(entry_point),
      std::move(abi), dispatch);
  return make_gfx_custom_kernel_stage_manifest(
      family, GfxKernelBackendDomain::OpenCl, GfxKernelStorageKind::Buffer,
      std::move(specialization_key), std::move(custom));
}

GfxOpenClSourceArtifact make_opencl_source_artifact(
    GfxKernelStageManifest manifest, std::string source_id, std::string source,
    std::vector<GfxOpenClSourceScalarArg> scalar_args,
    std::vector<size_t> direct_input_indices, GfxOpenClArtifactOp op,
    GfxOpenClArtifactInputMode input_mode, float scalar_constant_f32,
    std::vector<uint32_t> static_u32_scalars,
    GfxOpenClSourceElementCountSource element_count_source,
    std::vector<float> static_f32_scalars) {
  GfxOpenClSourceArtifact artifact{};
  artifact.valid = manifest.valid;
  artifact.stage_manifest = std::move(manifest);
  artifact.artifact_ref = make_gfx_kernel_artifact_ref(artifact.stage_manifest);
  artifact.artifact_ref.source_id = std::move(source_id);
  artifact.artifact_ref.entry_point =
      artifact.stage_manifest.custom_kernel.entry_point;
  artifact.source = std::move(source);
  artifact.scalar_args = std::move(scalar_args);
  artifact.static_u32_scalars = std::move(static_u32_scalars);
  artifact.static_f32_scalars = std::move(static_f32_scalars);
  artifact.direct_input_indices = std::move(direct_input_indices);

  const auto roles = materialize_gfx_kernel_external_buffer_roles(
      artifact.stage_manifest.custom_kernel.external_buffer_abi);
  artifact.arg_count = static_cast<uint32_t>(roles.size());
  artifact.direct_input_count =
      static_cast<uint32_t>(artifact.direct_input_indices.size());
  artifact.direct_output_count = 0;
  for (const auto role : roles) {
    if (role == GfxKernelBufferRole::TensorOutput) {
      ++artifact.direct_output_count;
    }
  }

  artifact.element_count_source = element_count_source;
  artifact.op = op;
  artifact.input_mode = input_mode;
  artifact.scalar_constant_f32 = scalar_constant_f32;
  artifact.valid =
      artifact.valid && artifact.artifact_ref.valid &&
      artifact.artifact_ref.kind == GfxKernelArtifactKind::OpenClSource &&
      !artifact.source.empty();
  return artifact;
}

GfxOpenClSourceArtifactPayload::GfxOpenClSourceArtifactPayload(
    GfxOpenClSourceArtifact artifact)
    : m_artifact(std::move(artifact)) {}

KernelArtifactPayloadKind
GfxOpenClSourceArtifactPayload::payload_kind() const noexcept {
  return KernelArtifactPayloadKind::OpenClSource;
}

std::string_view
GfxOpenClSourceArtifactPayload::backend_domain() const noexcept {
  return "opencl";
}

std::string_view GfxOpenClSourceArtifactPayload::source_id() const noexcept {
  return m_artifact.artifact_ref.source_id;
}

std::string_view GfxOpenClSourceArtifactPayload::entry_point() const noexcept {
  return m_artifact.artifact_ref.entry_point;
}

bool GfxOpenClSourceArtifactPayload::valid() const noexcept {
  if (!m_artifact.valid || !m_artifact.artifact_ref.valid ||
      m_artifact.artifact_ref.kind != GfxKernelArtifactKind::OpenClSource ||
      m_artifact.artifact_ref.backend_domain != GfxKernelBackendDomain::OpenCl ||
      m_artifact.source.empty()) {
    return false;
  }
  const bool expects_chunks =
      m_artifact.input_chunk_size != 0 || m_artifact.output_chunk_size != 0;
  if (!expects_chunks) {
    return m_artifact.planned_chunks.empty();
  }
  if (m_artifact.planned_chunks.empty()) {
    return false;
  }
  for (const auto &chunk : m_artifact.planned_chunks) {
    if (chunk.binding_count == 0 || chunk.element_count_multiplier == 0 ||
        chunk.element_count_divisor == 0 || !chunk.artifact ||
        !chunk.artifact->valid || !chunk.artifact->artifact_ref.valid ||
        chunk.artifact->artifact_ref.kind != GfxKernelArtifactKind::OpenClSource ||
        chunk.artifact->artifact_ref.backend_domain !=
            GfxKernelBackendDomain::OpenCl ||
        chunk.artifact->source.empty() ||
        !chunk.artifact->planned_chunks.empty()) {
      return false;
    }
    if (chunk.binding_role == GfxOpenClSourceChunkBindingRole::DirectInputs) {
      if (chunk.binding_begin + chunk.binding_count >
              m_artifact.direct_input_count ||
          chunk.artifact->direct_input_count != chunk.binding_count ||
          chunk.artifact->direct_output_count !=
              m_artifact.direct_output_count) {
        return false;
      }
      continue;
    }
    if (chunk.binding_role == GfxOpenClSourceChunkBindingRole::DirectOutputs) {
      if (chunk.binding_begin + chunk.binding_count >
              m_artifact.direct_output_count ||
          chunk.artifact->direct_input_count != m_artifact.direct_input_count ||
          chunk.artifact->direct_output_count != chunk.binding_count) {
        return false;
      }
      continue;
    }
    return false;
  }
  return true;
}

std::string gfx_opencl_source_artifact_build_options(
    const GfxOpenClSourceArtifact &artifact) {
  std::string joined;
  for (const auto &option : artifact.build_options) {
    if (!joined.empty()) {
      joined.push_back(' ');
    }
    joined += option;
  }
  return joined;
}

} // namespace gfx_plugin
} // namespace ov
