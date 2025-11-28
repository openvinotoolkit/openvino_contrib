// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "graph/mps_node_context.hpp"
#include "openvino/op/matmul.hpp"

namespace ov::metal_plugin::ops {

MetalNode* build_matmul(NodeContext& ctx, const ov::op::v0::MatMul& node);

}  // namespace ov::metal_plugin::ops
