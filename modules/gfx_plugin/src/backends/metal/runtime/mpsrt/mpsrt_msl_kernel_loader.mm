// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/runtime/mpsrt/mpsrt_msl_kernel_loader.hpp"

#include <string_view>

#include "backends/metal/runtime/mpsrt/mpsrt_context.hpp"

namespace ov {
namespace gfx_plugin {
namespace metal {
namespace mpsrt {

namespace runtime_mpsrt = ::ov::gfx_plugin::mpsrt;

namespace {

bool set_error(std::string* error, std::string_view message) {
    if (error) {
        *error = std::string(message);
    }
    return false;
}

bool validate_msl_source_descriptor(const GfxKernelSource& source, std::string* error) {
    if (!gfx_kernel_source_valid(source)) {
        return set_error(error, "GFX MPSRT: Metal kernel source descriptor is invalid");
    }
    if (source.source_language != GfxKernelSourceLanguage::MetalShadingLanguage) {
        return set_error(error, "GFX MPSRT: kernel source descriptor is not Metal Shading Language");
    }
    if (std::string_view(source.backend_domain) != std::string_view("apple_msl")) {
        return set_error(error, "GFX MPSRT: Metal kernel source descriptor has unsupported backend domain");
    }
    return true;
}

}  // namespace

runtime_mpsrt::MpsrtRuntimeStage MpsrtMslKernelLoader::make_stage(
    const GfxKernelSource& source,
    uint32_t threads_per_threadgroup,
    uint32_t flags) {
    runtime_mpsrt::MpsrtRuntimeStage stage;
    stage.kind = GfxMpsrtStageKind::MSLDispatch;
    stage.stage_record_key = source.kernel_id ? source.kernel_id : "";
    stage.dispatch_entry_point = source.entry_point ? source.entry_point : "";
    stage.dispatch_threads_per_threadgroup = threads_per_threadgroup;
    stage.dispatch_flags = flags;
    return stage;
}

id<MTLComputePipelineState> MpsrtMslKernelLoader::load_pipeline(
    MpsrtContext& context,
    const GfxKernelSource& source,
    uint32_t threads_per_threadgroup,
    bool& cache_hit,
    std::string* error,
    uint32_t flags) {
    cache_hit = false;
    if (!validate_msl_source_descriptor(source, error)) {
        return nil;
    }

    const auto stage = make_stage(source, threads_per_threadgroup, flags);
    return context.get_or_create_pipeline(stage, source.source, cache_hit, error);
}

}  // namespace mpsrt
}  // namespace metal
}  // namespace gfx_plugin
}  // namespace ov
