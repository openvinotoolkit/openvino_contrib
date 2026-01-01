// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <vector>

#include "openvino/core/node.hpp"

namespace ov {
namespace gfx_plugin {

struct KernelInputMapping {
    size_t func_inputs = 0;
    std::vector<size_t> kernel_inputs;
};

KernelInputMapping build_kernel_inputs(const std::shared_ptr<const ov::Node>& node,
                                       size_t func_inputs,
                                       const char* stage_name);

}  // namespace gfx_plugin
}  // namespace ov
