// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {
class DecomposeSingleSwish: public ngraph::pass::MatcherPass {
public:
    DecomposeSingleSwish();
};

class  DecomposeSwishWithBeta: public ngraph::pass::MatcherPass {
public:
    DecomposeSwishWithBeta();
};

class DecomposeSwish: public ngraph::pass::GraphRewrite {
public:
    DecomposeSwish() {
        add_matcher<DecomposeSingleSwish>();
        add_matcher<DecomposeSwishWithBeta>();
    }
};
}  // namespace pass
}  // namespace ArmPlugin
