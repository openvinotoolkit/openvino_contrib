// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {

class ConvertBatchNormLayout : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertBatchNormLayout", "0");
    ConvertBatchNormLayout();
};

class ConvertBatchToSpaceLayout : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertBatchToSpaceLayout", "0");
    ConvertBatchToSpaceLayout();
};

class ConvertDepthToSpaceLayout : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertDepthToSpaceLayout", "0");
    ConvertDepthToSpaceLayout();
};

class ConvertLayout: public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("ConvertLayout", "0");
    ConvertLayout() {
        add_matcher<ConvertBatchNormLayout>();
        add_matcher<ConvertBatchToSpaceLayout>();
        add_matcher<ConvertDepthToSpaceLayout>();
    }
};

}  // namespace pass
}  // namespace ArmPlugin
