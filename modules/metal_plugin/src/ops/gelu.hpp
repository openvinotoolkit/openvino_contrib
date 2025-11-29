// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/gelu.hpp"

#include "graph/mps_node_context.hpp"

namespace ov {
namespace metal_plugin {
namespace ops {

MetalNode* build_gelu(NodeContext& ctx, const ov::op::v7::Gelu& node);
MetalNode* build_gelu(NodeContext& ctx, const ov::op::v0::Gelu& node);

}  // namespace ops
}  // namespace metal_plugin
}  // namespace ov
