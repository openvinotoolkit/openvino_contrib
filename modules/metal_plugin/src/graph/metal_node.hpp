// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace metal_plugin {

class MPSGraphTensor;

enum class MetalOpType {
    Parameter,
    Constant,
    Result,
    Add,
    Relu,
    MatMul,
    Convolution,
    MaxPool,
    AvgPool,
    Softmax,
    BatchNorm,
    Tanh,
    Sigmoid,
    Elu,
    LeakyRelu,
    Gelu,
    LayerNorm,
};

enum class Layout { NCHW, NHWC };

struct TensorDesc {
    ov::Shape shape;
    ov::element::Type element_type;
    Layout layout = Layout::NCHW;
};

struct Value {
    TensorDesc desc;
    MPSGraphTensor* tensor = nullptr;
};

struct MetalOpDesc {
    MetalOpType type;
    std::string friendly_name;
    std::shared_ptr<const ov::Node> ov_node;
    std::vector<size_t> input_indices;

    // Convolution-specific
    ov::Strides strides;
    ov::Strides dilations;
    ov::CoordinateDiff pads_begin;
    ov::CoordinateDiff pads_end;
    size_t groups = 1;
    bool exclude_pad = false;
    bool fused_relu = false;
};

struct MetalNode {
    size_t index = 0;
    MetalOpDesc op;

    TensorDesc output_desc;
    std::vector<MetalNode*> deps;
    std::vector<MetalNode*> users;

    MPSGraphTensor* mps_tensor = nullptr;
};

}  // namespace metal_plugin
}  // namespace ov
