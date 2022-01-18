// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {

class ConvertLogicalBase : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    template <class T>
    ngraph::matcher_pass_callback convert_logical();
};

class ConvertLogicalNot: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertLogicalNot();
};

class ConvertLogicalAnd: public ConvertLogicalBase {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertLogicalAnd();
};

class ConvertLogicalOr: public ConvertLogicalBase {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertLogicalOr();
};

class ConvertLogicalXor: public ConvertLogicalBase {
public:
    NGRAPH_RTTI_DECLARATION;
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
