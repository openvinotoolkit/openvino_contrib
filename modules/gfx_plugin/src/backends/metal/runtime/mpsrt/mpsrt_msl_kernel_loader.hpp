// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#import <Metal/Metal.h>

#include <cstdint>
#include <string>

#include "kernel_ir/gfx_kernel_source.hpp"
#include "backends/metal/runtime/mpsrt/gfx_mpsrt_model.hpp"

namespace ov {
namespace gfx_plugin {
namespace metal {
namespace mpsrt {

class MpsrtContext;

class MpsrtMslKernelLoader final {
public:
    static ::ov::gfx_plugin::mpsrt::MpsrtRuntimeStage make_stage(
        const GfxKernelSource& source,
        uint32_t threads_per_threadgroup,
        uint32_t flags = GfxMpsrtMslDispatchFlagNone);

    static id<MTLComputePipelineState> load_pipeline(
        MpsrtContext& context,
        const GfxKernelSource& source,
        uint32_t threads_per_threadgroup,
        bool& cache_hit,
        std::string* error,
        uint32_t flags = GfxMpsrtMslDispatchFlagNone);
};

}  // namespace mpsrt
}  // namespace metal
}  // namespace gfx_plugin
}  // namespace ov
