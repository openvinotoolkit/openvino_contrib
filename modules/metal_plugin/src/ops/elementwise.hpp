// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "graph/mps_node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/relu.hpp"

namespace ov::metal_plugin::ops {

MetalNode* build_add(NodeContext& ctx, const ov::op::v1::Add& node);
MetalNode* build_relu(NodeContext& ctx, const ov::op::v0::Relu& node);

}  // namespace ov::metal_plugin::ops
