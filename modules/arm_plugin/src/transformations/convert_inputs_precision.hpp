// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "opset/opset.hpp"

namespace ArmPlugin {
namespace pass {

class ConvertPrecisionBase : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    template <class T>
    ngraph::matcher_pass_callback convert_precision(const std::vector<int>& indices);
};

class ConvertPReluPrecision: public ConvertPrecisionBase {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertPReluPrecision();
};

class ConvertProposalPrecision: public ConvertPrecisionBase {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertProposalPrecision();
};

class ConvertInterpolatePrecision: public ConvertPrecisionBase {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertInterpolatePrecision();
};

class AlignNodePrecision: public ngraph::pass::GraphRewrite {
public:
    AlignNodePrecision() {
        add_matcher<ConvertPReluPrecision>();
        add_matcher<ConvertProposalPrecision>();
    }
};
}  // namespace pass
}  // namespace ArmPlugin
