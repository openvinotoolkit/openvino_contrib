// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace metal_plugin {
namespace transforms {

inline constexpr const char kMetalFusePrevConvAttr[] = "METAL_FUSE_PREV_CONV";

// Marks Relu nodes that can be fused into a preceding Convolution for the METAL backend.
class ConvReluFusion : public ov::pass::MatcherPass {
public:
    ConvReluFusion() {
        auto conv_pattern = ov::pass::pattern::wrap_type<ov::op::v1::Convolution>();
        auto relu_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({conv_pattern});

        matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
            auto relu = std::dynamic_pointer_cast<ov::op::v0::Relu>(m.get_match_root());
            if (!relu) {
                return false;
            }
            auto conv = std::dynamic_pointer_cast<ov::op::v1::Convolution>(relu->get_input_node_shared_ptr(0));
            if (!conv) {
                return false;
            }
            if (conv->get_output_size() != 1) {
                return false;
            }
            if (!conv->output(0).get_target_inputs().empty() &&
                conv->output(0).get_target_inputs().size() != 1) {
                return false;
            }
            auto& rt_info = relu->get_rt_info();
            rt_info[kMetalFusePrevConvAttr] = true;
            return true;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(relu_pattern, "MetalConvReluFusion");
        register_matcher(m, std::move(callback));
    }
};

}  // namespace transforms
}  // namespace metal_plugin
}  // namespace ov
