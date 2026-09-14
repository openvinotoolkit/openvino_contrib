// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <vector>

#include "backends/metal/common/mpsrt/gfx_mpsrt_abi.hpp"
#include "backends/metal/common/mpsrt/gfx_mpsrt_program.hpp"
#include "kernel_ir/gfx_kernel_manifest.hpp"

namespace ov {
namespace gfx_plugin {

enum class GfxAppleMpsVendorPrimitiveKind {
  None,
  Gemm,
  Conv2D,
  Pool2D,
  Resize2D,
  Softmax,
  TopK,
  Sdpa,
};

struct GfxAppleMpsVendorPrimitiveDescriptor {
  GfxAppleMpsVendorPrimitiveKind kind = GfxAppleMpsVendorPrimitiveKind::None;
  GfxMpsrtGemmAbiDesc gemm{};
  GfxMpsrtConv2DAbiDesc conv2d{};
  GfxMpsrtPool2DAbiDesc pool2d{};
  GfxMpsrtResize2DAbiDesc resize2d{};
  GfxMpsrtSoftmaxAbiDesc softmax{};
  GfxMpsrtTopKAbiDesc topk{};
  GfxMpsrtSdpaAbiDesc sdpa{};
};

struct GfxAppleMpsVendorPrimitiveContract {
  bool valid = false;
  GfxAppleMpsVendorPrimitiveDescriptor descriptor;
  std::vector<GfxKernelBufferRole> semantic_input_roles;
  GfxMpsrtExternalBufferAbiPlan external_buffer_abi{};
  std::vector<GfxMpsrtTensorDesc> input_descs;
  std::vector<GfxMpsrtTensorDesc> output_descs;
};

} // namespace gfx_plugin
} // namespace ov
