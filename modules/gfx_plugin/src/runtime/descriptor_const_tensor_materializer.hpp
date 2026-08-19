// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string_view>
#include <vector>

#include "runtime/gpu_tensor.hpp"

namespace ov {
namespace gfx_plugin {

class GpuBufferManager;
struct RuntimeStageExecutableDescriptor;

struct DescriptorConstTensorSlots {
  std::vector<GpuTensor> buffers;
  std::vector<bool> present;
};

DescriptorConstTensorSlots materialize_descriptor_const_tensor_slots(
    GpuBufferManager &buffer_manager,
    const RuntimeStageExecutableDescriptor &descriptor,
    std::string_view cache_prefix);

std::vector<GpuTensor *> descriptor_const_tensor_args(
    DescriptorConstTensorSlots &slots, size_t expected_count);

} // namespace gfx_plugin
} // namespace ov
