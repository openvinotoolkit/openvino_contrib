// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "graph/mps_node_context.hpp"

#if __has_include("openvino/op/layer_norm.hpp")
#define OV_LAYER_NORM_AVAILABLE 1
#include "openvino/op/layer_norm.hpp"
#else
#define OV_LAYER_NORM_AVAILABLE 0
#endif

namespace ov {
namespace metal_plugin {
namespace ops {

#if OV_LAYER_NORM_AVAILABLE
MetalNode* build_layer_norm(NodeContext& ctx, const ov::op::v12::LayerNorm& node);
#endif

}  // namespace ops
}  // namespace metal_plugin
}  // namespace ov
