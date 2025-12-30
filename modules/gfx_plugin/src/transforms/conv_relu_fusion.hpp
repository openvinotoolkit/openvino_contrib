// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace gfx_plugin {
namespace transforms {

inline constexpr const char kGfxFusePrevConvAttr[] = "GFX_FUSE_PREV_CONV";

// Marks Relu nodes that can be fused into a preceding Convolution for the GFX backend.
class ConvReluFusion : public ov::pass::MatcherPass {
public:
    ConvReluFusion();
};

}  // namespace transforms
}  // namespace gfx_plugin
}  // namespace ov
