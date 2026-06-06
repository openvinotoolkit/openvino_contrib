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
    GfxOpenClSourceArtifact artifact,
    std::shared_ptr<const ov::Node> source_node) const {
  verify_opencl_source_descriptor(descriptor, artifact);
  return create_opencl_source_stage(m_context, descriptor, std::move(artifact),
                                    std::move(source_node));
}

} // namespace gfx_plugin
} // namespace ov
