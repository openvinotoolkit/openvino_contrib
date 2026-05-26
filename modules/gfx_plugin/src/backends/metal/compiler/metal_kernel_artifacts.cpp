// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/compiler/metal_kernel_artifacts.hpp"

#include <memory>
#include <string>
#include <utility>

#include "kernel_ir/gfx_kernel_source.hpp"
#include "mlir/msl_codegen_apple_msl_shape.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

constexpr const char* kMetalShapeOfMslKernelUnit = "metal/generated/shapeof";

std::shared_ptr<const KernelArtifactPayload> materialize_generated_msl_payload(
    KernelArtifactDescriptor& descriptor,
    GfxMslGeneratedKernelSourcePlan source_plan) {
    if (!source_plan.valid()) {
        return {};
    }
    descriptor.entry_point = source_plan.source.entry_point;
    descriptor.compile_options_key = "metal_msl_source";
    descriptor.abi_arg_count = source_plan.source.signature.arg_count;
    descriptor.abi_output_arg_count =
        source_plan.source.signature.output_arg_count;
    return std::make_shared<GfxKernelSourcePayload>(
        descriptor.kernel.kernel_id,
        descriptor.kernel.backend_domain,
        descriptor.entry_point,
        GfxKernelSourceLanguage::MetalShadingLanguage,
        std::move(source_plan.source.msl_source));
}

std::shared_ptr<const KernelArtifactPayload> resolve_metal_payload(
    KernelArtifactDescriptor& descriptor,
    const PlannedOperation& op) {
    if (descriptor.payload_kind != KernelArtifactPayloadKind::MslSource ||
        descriptor.kernel.backend_domain != "metal" ||
        !op.source_node) {
        return {};
    }

    if (descriptor.kernel.kernel_id == kMetalShapeOfMslKernelUnit) {
        return materialize_generated_msl_payload(
            descriptor,
            make_shapeof_msl_kernel_source_plan(op.source_node));
    }

    return {};
}

}  // namespace

KernelArtifactPayloadResolver make_metal_kernel_artifact_payload_resolver() {
    return [](KernelArtifactDescriptor& descriptor,
              const PlannedOperation& op) {
        return resolve_metal_payload(descriptor, op);
    };
}

}  // namespace compiler
}  // namespace gfx_plugin
}  // namespace ov
