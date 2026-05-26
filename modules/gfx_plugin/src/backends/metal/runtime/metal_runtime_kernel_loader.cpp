// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/runtime/metal_runtime_kernel_loader.hpp"

#include <algorithm>
#include <memory>
#include <string_view>

#include "kernel_ir/gfx_kernel_source.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

bool is_metal_source_domain(std::string_view domain) noexcept {
    return domain == "metal" || domain == "apple_msl";
}

bool is_source_origin(compiler::KernelArtifactOrigin origin) noexcept {
    return origin == compiler::KernelArtifactOrigin::Generated ||
           origin == compiler::KernelArtifactOrigin::HandwrittenException;
}

uint32_t descriptor_output_arg_count(
    const RuntimeStageExecutableDescriptor& descriptor) {
    if (descriptor.abi_output_arg_count != 0) {
        return descriptor.abi_output_arg_count;
    }
    return static_cast<uint32_t>(std::count(descriptor.tensor_roles.begin(),
                                            descriptor.tensor_roles.end(),
                                            "tensor_output"));
}

uint32_t descriptor_arg_count(
    const RuntimeStageExecutableDescriptor& descriptor) {
    if (descriptor.abi_arg_count != 0) {
        return descriptor.abi_arg_count;
    }
    return static_cast<uint32_t>(descriptor.tensor_roles.size() +
                                 descriptor.scalar_roles.size());
}

void verify_msl_source_descriptor(
    const RuntimeStageExecutableDescriptor& descriptor,
    const GfxKernelSourcePayload& payload) {
    OPENVINO_ASSERT(descriptor.payload_kind ==
                        compiler::KernelArtifactPayloadKind::MslSource,
                    "GFX Metal: runtime descriptor payload is not MSL source");
    OPENVINO_ASSERT(is_metal_source_domain(descriptor.backend_domain),
                    "GFX Metal: MSL descriptor backend domain drift: ",
                    descriptor.backend_domain);
    OPENVINO_ASSERT(is_source_origin(descriptor.origin),
                    "GFX Metal: MSL descriptor origin must be generated or "
                    "handwritten exception");
    OPENVINO_ASSERT(!descriptor.manifest_ref.empty() &&
                        !descriptor.abi_fingerprint.empty() &&
                        !descriptor.artifact_key.empty(),
                    "GFX Metal: MSL descriptor identity is incomplete");

    const auto& source = payload.source();
    OPENVINO_ASSERT(source.source_language ==
                        GfxKernelSourceLanguage::MetalShadingLanguage,
                    "GFX Metal: runtime loader received non-MSL source "
                    "payload");
    OPENVINO_ASSERT(payload.payload_kind() ==
                        compiler::KernelArtifactPayloadKind::MslSource,
                    "GFX Metal: payload kind drift for MSL descriptor");
    OPENVINO_ASSERT(payload.backend_domain() == descriptor.backend_domain,
                    "GFX Metal: payload backend domain drift for ",
                    descriptor.kernel_id);
    OPENVINO_ASSERT(payload.source_id() == descriptor.kernel_id,
                    "GFX Metal: payload source id drift for ",
                    descriptor.kernel_id);
    OPENVINO_ASSERT(payload.entry_point() == descriptor.entry_point,
                    "GFX Metal: payload entry point drift for ",
                    descriptor.kernel_id);
    OPENVINO_ASSERT(descriptor_arg_count(descriptor) != 0 &&
                        descriptor_output_arg_count(descriptor) != 0,
                    "GFX Metal: MSL descriptor must carry explicit ABI roles "
                    "for ",
                    descriptor.kernel_id);
}

}  // namespace

bool MetalRuntimeKernelLoader::has_msl_source_payload(
    const RuntimeStageExecutableDescriptor& descriptor) noexcept {
    return descriptor.payload_kind == compiler::KernelArtifactPayloadKind::MslSource &&
           static_cast<bool>(descriptor.payload);
}

KernelSource MetalRuntimeKernelLoader::load_msl_source(
    const RuntimeStageExecutableDescriptor& descriptor) {
    OPENVINO_ASSERT(descriptor.payload,
                    "GFX Metal: MSL source descriptor is missing payload");
    auto payload =
        std::dynamic_pointer_cast<const GfxKernelSourcePayload>(descriptor.payload);
    OPENVINO_ASSERT(payload,
                    "GFX Metal: MSL source payload has unexpected type for ",
                    descriptor.kernel_id);
    OPENVINO_ASSERT(payload->valid(),
                    "GFX Metal: MSL source payload is invalid for ",
                    descriptor.kernel_id);
    verify_msl_source_descriptor(descriptor, *payload);

    const auto& source = payload->source();
    KernelSource kernel_source;
    kernel_source.entry_point = source.entry_point;
    kernel_source.msl_source = source.source;
    kernel_source.signature.arg_count = descriptor_arg_count(descriptor);
    kernel_source.signature.output_arg_count =
        descriptor_output_arg_count(descriptor);
    return kernel_source;
}

}  // namespace gfx_plugin
}  // namespace ov
