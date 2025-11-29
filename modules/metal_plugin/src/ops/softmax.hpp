// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "graph/mps_node_context.hpp"
#include "openvino/op/softmax.hpp"

namespace ov::metal_plugin::ops {

MetalNode* build_softmax(NodeContext& ctx, const ov::op::v1::Softmax& node);
MetalNode* build_softmax(NodeContext& ctx, const ov::op::v8::Softmax& node);

}  // namespace ov::metal_plugin::ops
