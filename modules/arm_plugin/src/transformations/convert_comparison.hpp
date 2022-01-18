// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {

class ConvertComparisionBase : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    template <class T>
    ngraph::matcher_pass_callback convert_comparision();
};

class ConvertEqual: public ConvertComparisionBase {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertEqual();
};

class ConvertNotEqual: public ConvertComparisionBase {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertNotEqual();
};

class ConvertGreater: public ConvertComparisionBase {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertGreater();
};

class ConvertGreaterEqual: public ConvertComparisionBase {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertGreaterEqual();
};

class ConvertLess: public ConvertComparisionBase {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertLess();
};

class ConvertLessEqual: public ConvertComparisionBase {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertLessEqual();
};

class ConvertComparison: public ngraph::pass::GraphRewrite {
public:
    ConvertComparison() {
        add_matcher<ConvertEqual>();
        add_matcher<ConvertNotEqual>();
        add_matcher<ConvertGreater>();
        add_matcher<ConvertGreaterEqual>();
        add_matcher<ConvertLess>();
        add_matcher<ConvertLessEqual>();
    }
};
}  // namespace pass
}  // namespace ArmPlugin
