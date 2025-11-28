// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace metal_plugin {

// Execute MPSGraph with host-side OpenVINO tensors (synchronous).
// Inputs/outputs sizes must match input_tensors/output_tensors from build_mps_graph.
void mps_execute(const std::shared_ptr<void>& graph,
                 const std::vector<void*>& input_tensors,
                 const std::vector<void*>& output_tensors,
                 const std::vector<ov::Tensor>& inputs,
                 std::vector<ov::Tensor>& outputs);

}  // namespace metal_plugin
}  // namespace ov
