// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "graph/mps_node_context.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/max_pool.hpp"

namespace ov::metal_plugin::ops {

MetalNode* build_max_pool(NodeContext& ctx,
                          const ov::op::v1::MaxPool& node);

MetalNode* build_avg_pool(NodeContext& ctx,
                          const ov::op::v1::AvgPool& node);

}  // namespace ov::metal_plugin::ops
