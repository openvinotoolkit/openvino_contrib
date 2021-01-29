// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {
class ConvertLogicalBase : public ngraph::pass::MatcherPass {
public:
    template <class T>
    ngraph::matcher_pass_callback convert_logical();
};

class ConvertLogicalNot: public ngraph::pass::MatcherPass {
public:
    ConvertLogicalNot();
};

class ConvertLogicalAnd: public ConvertLogicalBase {
public:
    ConvertLogicalAnd();
};

class ConvertLogicalOr: public ConvertLogicalBase {
public:
    ConvertLogicalOr();
};

class ConvertLogicalXor: public ConvertLogicalBase {
public:
    ConvertLogicalXor();
};

class ConvertLogical: public ngraph::pass::GraphRewrite {
public:
    ConvertLogical() {
        add_matcher<ConvertLogicalNot>();
        add_matcher<ConvertLogicalAnd>();
        add_matcher<ConvertLogicalOr>();
        add_matcher<ConvertLogicalXor>();
    }
};
}  // namespace pass
}  // namespace ArmPlugin
