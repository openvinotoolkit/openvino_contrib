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

}  // namespace metal_plugin
}  // namespace ov
