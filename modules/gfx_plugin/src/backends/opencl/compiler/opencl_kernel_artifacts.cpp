// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/compiler/opencl_kernel_artifacts.hpp"

#include <memory>
#include <utility>

#include "kernel_ir/gfx_opencl_source_artifacts.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

std::shared_ptr<const KernelArtifactPayload>
resolve_opencl_payload(KernelArtifactDescriptor &descriptor,
                       const PlannedOperation &op) {
  if (descriptor.kernel.backend_domain != "opencl" ||
      descriptor.payload_kind != KernelArtifactPayloadKind::OpenClSource ||
      !op.source_node) {
    return {};
  }

  auto source_artifact = resolve_gfx_opencl_source_artifact(op.source_node);
  if (!source_artifact || !source_artifact->valid) {
    return {};
  }

  descriptor.entry_point = source_artifact->artifact_ref.entry_point;
  descriptor.compile_options_key =
      gfx_opencl_source_artifact_build_options(*source_artifact);
  descriptor.abi_arg_count = source_artifact->arg_count;
  descriptor.abi_output_arg_count = source_artifact->direct_output_count;
  return std::make_shared<GfxOpenClSourceArtifactPayload>(
      std::move(*source_artifact));
}

} // namespace

KernelArtifactPayloadResolver make_opencl_kernel_artifact_payload_resolver() {
  return [](KernelArtifactDescriptor &descriptor, const PlannedOperation &op) {
    return resolve_opencl_payload(descriptor, op);
  };
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
