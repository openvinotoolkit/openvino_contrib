// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "kernel_ir/gfx_kernel_manifest.hpp"
#include "runtime/gfx_mpsrt_abi.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace ov {
namespace gfx_plugin {

enum class GfxMslKernelFamily {
    Unknown = 0,
    EltwiseFusedBuffer = 1,
    TransposePackND = 2,
    ConcatSplitGeneric = 3,
    GatherScatterIndexed = 4,
    RmsnormRopeFused = 5,
    MaskedSoftmaxAttention = 6,
    KvCacheUpdate = 7,
    Conv3DDirectOrIm2col = 8,
    ReductionBuffer = 9,
    Conv2DDirectOrIm2col = 10,
};

using GfxMslExternalBufferAbiSpec = GfxKernelExternalBufferAbiSpec;

struct GfxMslKernelPlan {
    bool valid = false;
    GfxMslKernelFamily family = GfxMslKernelFamily::Unknown;
    std::string family_name;
    std::string required_entry_point;
    uint32_t abi_kernel_family = 0;
    uint32_t dispatch_flags = GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired;
    GfxMpsrtStorage storage = GfxMpsrtStorage::Buffer;
    GfxMpsrtLayout layout = GfxMpsrtLayout::Linear;
    uint32_t threads_per_threadgroup = 256;
    bool precompiled_metallib_required = true;
    GfxMslExternalBufferAbiSpec external_buffer_abi;
    GfxKernelCustomManifest kernel_manifest;
    GfxKernelStageManifest stage_manifest;
};

const char* gfx_msl_kernel_family_name(GfxMslKernelFamily family);
const char* gfx_msl_required_kernel_entry_point(GfxMslKernelFamily family);
uint32_t gfx_msl_kernel_family_abi_id(GfxMslKernelFamily family);
GfxMslExternalBufferAbiSpec gfx_msl_external_buffer_abi_spec(GfxMslKernelFamily family);
GfxMslExternalBufferAbiSpec gfx_msl_external_buffer_abi_spec(std::string_view stage_type,
                                                             std::string_view entry_point,
                                                             GfxMslKernelFamily family);
GfxMslKernelFamily classify_msl_kernel_family(std::string_view stage_type,
                                              std::string_view entry_point = {});
GfxMslKernelPlan make_msl_kernel_plan(std::string_view stage_type,
                                      std::string_view entry_point = {});

}  // namespace gfx_plugin
}  // namespace ov
