// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/batch_norm.hpp"

#include "graph/mps_node_context.hpp"

namespace ov {
namespace metal_plugin {
namespace ops {

MetalNode* build_batch_norm(NodeContext& ctx, const ov::op::v5::BatchNormInference& node);
MetalNode* build_batch_norm(NodeContext& ctx, const ov::op::v0::BatchNormInference& node);

}  // namespace ops
}  // namespace metal_plugin
}  // namespace ov
