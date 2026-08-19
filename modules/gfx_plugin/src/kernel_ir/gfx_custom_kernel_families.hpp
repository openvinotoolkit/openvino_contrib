// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "kernel_ir/gfx_kernel_manifest.hpp"

#include <cstdint>
#include <string_view>

namespace ov {
namespace gfx_plugin {

enum class GfxKernelFamily {
  Unknown = 0,
  EltwiseFusedBuffer = 1,
  TransposePackND = 2,
  ConcatSplitGeneric = 3,
  GatherScatterIndexed = 4,
  RmsnormRopeFused = 5,
  MaskedSoftmaxAttention = 6,
  KvCacheUpdate = 7,
  Conv3DDirect = 8,
  ReductionBuffer = 9,
  Conv2DDirect = 10,
  MatMulBuffer = 11,
  Pool2DWindow = 12,
  BatchNormBuffer = 13,
  SoftmaxBuffer = 14,
};

struct GfxCustomKernelStagePlan {
  bool valid = false;
  GfxKernelFamily family = GfxKernelFamily::Unknown;
  GfxKernelStageManifest stage_manifest;
};

const char *gfx_kernel_family_name(GfxKernelFamily family);
const char *gfx_kernel_required_entry_point(GfxKernelFamily family);
uint32_t gfx_kernel_family_abi_id(GfxKernelFamily family);
GfxKernelStageFamily
gfx_kernel_stage_family_from_kernel_family(GfxKernelFamily family);
GfxKernelExternalBufferAbiSpec
gfx_kernel_external_buffer_abi_spec_for_family(GfxKernelFamily family);
GfxKernelExternalBufferAbiSpec
gfx_kernel_external_buffer_abi_spec_for_stage(std::string_view stage_type,
                                              std::string_view entry_point,
                                              GfxKernelFamily family);
GfxKernelDispatchPolicy
gfx_kernel_dispatch_policy_for_family(GfxKernelFamily family);
GfxKernelDispatchPolicy
gfx_kernel_dispatch_policy_for_stage(std::string_view stage_type,
                                     std::string_view entry_point,
                                     GfxKernelFamily family);
GfxKernelFamily
classify_gfx_custom_kernel_family(std::string_view stage_type,
                                  std::string_view entry_point = {});
GfxCustomKernelStagePlan make_gfx_custom_kernel_stage_plan(
    std::string_view stage_type, std::string_view entry_point,
    GfxKernelBackendDomain backend_domain,
    GfxKernelStorageKind storage = GfxKernelStorageKind::Buffer,
    std::string_view specialization_prefix = {});

} // namespace gfx_plugin
} // namespace ov
