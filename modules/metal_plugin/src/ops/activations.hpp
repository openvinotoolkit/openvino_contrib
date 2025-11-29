// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "openvino/op/tanh.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/prelu.hpp"

#include "graph/mps_node_context.hpp"

namespace ov {
namespace metal_plugin {
namespace ops {

MetalNode* build_tanh(NodeContext& ctx, const ov::op::v0::Tanh& node);
MetalNode* build_sigmoid(NodeContext& ctx, const ov::op::v0::Sigmoid& node);
MetalNode* build_elu(NodeContext& ctx, const ov::op::v0::Elu& node);
MetalNode* build_leaky_relu(NodeContext& ctx, const ov::op::v0::PRelu& node);

}  // namespace ops
}  // namespace metal_plugin
}  // namespace ov
