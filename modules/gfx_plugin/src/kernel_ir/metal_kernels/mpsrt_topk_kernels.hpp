// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "kernel_ir/gfx_kernel_source.hpp"

namespace ov {
namespace gfx_plugin {

enum class MpsrtTopKValueType {
    F32,
    F16,
};

const GfxKernelSource& mpsrt_topk_pack_u32_to_i64_kernel_source();
const GfxKernelSource& mpsrt_topk_stable_i64_indices_kernel_source(MpsrtTopKValueType value_type);

}  // namespace gfx_plugin
}  // namespace ov
