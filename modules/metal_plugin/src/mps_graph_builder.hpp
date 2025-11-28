// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "openvino/core/model.hpp"

namespace ov {
namespace metal_plugin {

enum class GraphLayout { NHWC, NCHW };

struct MPSGraphBuildResult {
    std::shared_ptr<void> graph;                 // retained MPSGraph*
    std::vector<void*> input_tensors;            // MPSGraphTensor*
    std::vector<void*> output_tensors;           // MPSGraphTensor*
    GraphLayout internal_layout = GraphLayout::NHWC;
};

// Builds MPSGraph from an OpenVINO model. Only a minimal subset is supported so far.
MPSGraphBuildResult build_mps_graph(const std::shared_ptr<const ov::Model>& model,
                                    GraphLayout layout = GraphLayout::NHWC);

// Execute MPSGraph with host-side OpenVINO tensors (synchronous).
// Inputs/outputs sizes must match input_tensors/output_tensors from build_mps_graph.
void mps_execute(const std::shared_ptr<void>& graph,
                 const std::vector<void*>& input_tensors,
                 const std::vector<void*>& output_tensors,
                 const std::vector<ov::Tensor>& inputs,
                 std::vector<ov::Tensor>& outputs);

}  // namespace metal_plugin
}  // namespace ov
