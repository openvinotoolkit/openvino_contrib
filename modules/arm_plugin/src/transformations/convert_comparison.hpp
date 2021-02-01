// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {

class ConvertComparisionBase : public ngraph::pass::MatcherPass {
public:
    template <class T>
    ngraph::matcher_pass_callback convert_comparision();
};

class ConvertEqual: public ConvertComparisionBase {
public:
    ConvertEqual();
};

class ConvertNotEqual: public ConvertComparisionBase {
public:
    ConvertNotEqual();
};

class ConvertGreater: public ConvertComparisionBase {
public:
    ConvertGreater();
};

class ConvertGreaterEqual: public ConvertComparisionBase {
public:
    ConvertGreaterEqual();
};

class ConvertLess: public ConvertComparisionBase {
public:
    ConvertLess();
};

class ConvertLessEqual: public ConvertComparisionBase {
public:
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
