// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/runtime/opencl_runtime_kernel_loader.hpp"

#include <utility>

#include "backends/opencl/runtime/opencl_source_stage.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

void verify_opencl_source_descriptor(
    const RuntimeStageExecutableDescriptor &descriptor,
    const GfxOpenClSourceArtifact &artifact) {
  OPENVINO_ASSERT(artifact.valid, "GFX OpenCL: source artifact is invalid");
  OPENVINO_ASSERT(artifact.artifact_ref.valid,
                  "GFX OpenCL: source artifact ref is invalid");
  OPENVINO_ASSERT(
      artifact.artifact_ref.kind == GfxKernelArtifactKind::OpenClSource,
      "GFX OpenCL: runtime loader received non-OpenCL source artifact");
  OPENVINO_ASSERT(
      artifact.artifact_ref.backend_domain == GfxKernelBackendDomain::OpenCl,
      "GFX OpenCL: runtime loader received non-OpenCL artifact domain");
  OPENVINO_ASSERT(
      descriptor.payload_kind == KernelArtifactPayloadKind::OpenClSource,
      "GFX OpenCL: runtime descriptor payload is not OpenCL source");
  OPENVINO_ASSERT(
      descriptor.backend_domain == "opencl",
      "GFX OpenCL: runtime descriptor backend domain must be opencl");
  OPENVINO_ASSERT(descriptor.origin == KernelArtifactOrigin::Generated ||
                      descriptor.origin ==
                          KernelArtifactOrigin::HandwrittenException,
                  "GFX OpenCL: source runtime descriptor must be generated or "
                  "handwritten exception");
  OPENVINO_ASSERT(descriptor.entry_point == artifact.artifact_ref.entry_point,
                  "GFX OpenCL: runtime descriptor entry point drift for ",
                  artifact.artifact_ref.entry_point);
  OPENVINO_ASSERT(descriptor.kernel_id == artifact.artifact_ref.source_id,
                  "GFX OpenCL: runtime descriptor source id drift for ",
                  artifact.artifact_ref.source_id);
  OPENVINO_ASSERT(!descriptor.manifest_ref.empty() &&
                      !descriptor.abi_fingerprint.empty() &&
                      !descriptor.artifact_key.empty(),
                  "GFX OpenCL: runtime descriptor identity is incomplete");
  OPENVINO_ASSERT(
      descriptor.launch_plan.valid &&
          !descriptor.launch_plan.buffer_roles.empty(),
      "GFX OpenCL: runtime descriptor launch plan is incomplete for ",
      descriptor.stage_name);
  OPENVINO_ASSERT(
      descriptor.launch_plan.buffer_roles.size() == descriptor.abi_arg_count,
      "GFX OpenCL: runtime descriptor launch-plan arg count drift for ",
      descriptor.stage_name);
  OPENVINO_ASSERT(
      descriptor.abi_arg_count == artifact.arg_count,
      "GFX OpenCL: runtime descriptor/source artifact arg count drift for ",
      descriptor.stage_name);
  OPENVINO_ASSERT(
      descriptor.abi_output_arg_count == artifact.direct_output_count,
      "GFX OpenCL: runtime descriptor/source artifact output count drift for ",
      descriptor.stage_name);
  OPENVINO_ASSERT(
      descriptor.launch_plan.scalar_arg_kinds.size() ==
          artifact.scalar_args.size(),
      "GFX OpenCL: runtime descriptor/source artifact scalar ABI drift for ",
      descriptor.stage_name);
  for (const auto &planned : artifact.planned_chunks) {
    OPENVINO_ASSERT(planned.binding_count != 0,
                    "GFX OpenCL: planned source dispatch has empty binding "
                    "range for ",
                    descriptor.stage_name);
    OPENVINO_ASSERT(planned.element_count_multiplier != 0 &&
                        planned.element_count_divisor != 0,
                    "GFX OpenCL: planned source dispatch has invalid "
                    "element-count scale for ",
                    descriptor.stage_name);
    OPENVINO_ASSERT(
        planned.artifact && planned.artifact->valid &&
            planned.artifact->artifact_ref.valid &&
            planned.artifact->artifact_ref.kind ==
                GfxKernelArtifactKind::OpenClSource &&
            planned.artifact->artifact_ref.backend_domain ==
                GfxKernelBackendDomain::OpenCl &&
            planned.artifact->planned_chunks.empty(),
        "GFX OpenCL: planned source dispatch artifact contract is invalid for ",
        descriptor.stage_name);
    if (planned.binding_role == GfxOpenClSourceChunkBindingRole::DirectInputs) {
      OPENVINO_ASSERT(planned.artifact->direct_input_count ==
                          planned.binding_count,
                      "GFX OpenCL: planned source dispatch input binding range "
                      "drift for ",
                      descriptor.stage_name);
      OPENVINO_ASSERT(planned.artifact->direct_output_count ==
                          artifact.direct_output_count,
                      "GFX OpenCL: planned source dispatch output ABI drift for ",
                      descriptor.stage_name);
    } else {
      OPENVINO_ASSERT(planned.binding_role ==
                          GfxOpenClSourceChunkBindingRole::DirectOutputs,
                      "GFX OpenCL: planned source dispatch binding role is "
                      "unknown for ",
                      descriptor.stage_name);
      OPENVINO_ASSERT(planned.artifact->direct_input_count ==
                          artifact.direct_input_count,
                      "GFX OpenCL: planned source dispatch input ABI drift for ",
                      descriptor.stage_name);
      OPENVINO_ASSERT(planned.artifact->direct_output_count ==
                          planned.binding_count,
                      "GFX OpenCL: planned source dispatch output binding "
                      "range drift for ",
                      descriptor.stage_name);
    }
  }
}

} // namespace

OpenClRuntimeKernelLoader::OpenClRuntimeKernelLoader(
    std::shared_ptr<OpenClRuntimeContext> context)
    : m_context(std::move(context)) {
  OPENVINO_ASSERT(m_context,
                  "GFX OpenCL: runtime kernel loader requires context");
}

std::unique_ptr<GpuStage> OpenClRuntimeKernelLoader::load_source_stage(
    const RuntimeStageExecutableDescriptor &descriptor,
    GfxOpenClSourceArtifact artifact) const {
  verify_opencl_source_descriptor(descriptor, artifact);
  return create_opencl_source_stage(m_context, descriptor, std::move(artifact));
}

} // namespace gfx_plugin
} // namespace ov
