// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {


class ConvertArmConvolutionLayout : public ov::pass::MatcherPass {
    public:
        OPENVINO_RTTI("ConvertArmConvolutionLayout", "0");
        ConvertArmConvolutionLayout();
};

class ConvertArmMaxPoolV1Layout : public ov::pass::MatcherPass {
    public:
        OPENVINO_RTTI("ConvertArmMaxPoolV1Layout", "0");
        ConvertArmMaxPoolV1Layout();
};

class ConvertArmMaxPoolV8Layout : public ov::pass::MatcherPass {
    public:
        OPENVINO_RTTI("ConvertArmMaxPoolV8Layout", "0");
        ConvertArmMaxPoolV8Layout();
};

class ConvertArmAvgPoolLayout : public ov::pass::MatcherPass {
    public:
        OPENVINO_RTTI("ConvertArmAvgPoolLayout", "0");
        ConvertArmAvgPoolLayout();
};

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

class ConvertLayout: public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("ConvertLayout", "0");
    ConvertLayout() {
        add_matcher<ConvertArmConvolutionLayout>();
        add_matcher<ConvertArmMaxPoolV1Layout>();
        add_matcher<ConvertArmMaxPoolV8Layout>();
        add_matcher<ConvertArmAvgPoolLayout>();
        add_matcher<ConvertBatchNormLayout>();
        add_matcher<ConvertBatchToSpaceLayout>();
    }
};

}  // namespace pass
}  // namespace ArmPlugin
