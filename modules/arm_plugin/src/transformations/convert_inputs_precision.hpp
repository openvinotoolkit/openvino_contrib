// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "opset/opset.hpp"

namespace ArmPlugin {
namespace pass {

class ConvertPrecisionBase : public ngraph::pass::MatcherPass {
public:
    template <class T>
    ngraph::matcher_pass_callback convert_precision(const std::vector<int>& indices);
};

class ConvertPReluPrecision: public ConvertPrecisionBase {
public:
    ConvertPReluPrecision();
};

class ConvertProposalPrecision: public ConvertPrecisionBase {
public:
    ConvertProposalPrecision();
};

class ConvertInterpolatePrecision: public ConvertPrecisionBase {
public:
    ConvertInterpolatePrecision();
};

class AlignNodePrecision: public ngraph::pass::GraphRewrite {
public:
    AlignNodePrecision() {
        add_matcher<ConvertPReluPrecision>();
        add_matcher<ConvertProposalPrecision>();
        add_matcher<ConvertInterpolatePrecision>();
    }
};
}  // namespace pass
}  // namespace ArmPlugin
