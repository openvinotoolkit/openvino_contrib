// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {

class ConvertLogicalBase : public ngraph::pass::MatcherPass {
public:
    OPENVINO_OP("ConvertLogicalBase");
    template <class T>
    ngraph::matcher_pass_callback convert_logical();
};

class ConvertLogicalNot: public ngraph::pass::MatcherPass {
public:
    OPENVINO_OP("ConvertLogicalNot");
    ConvertLogicalNot();
};

class ConvertLogicalAnd: public ConvertLogicalBase {
public:
    OPENVINO_OP("ConvertLogicalAnd");
    ConvertLogicalAnd();
};

class ConvertLogicalOr: public ConvertLogicalBase {
public:
    OPENVINO_OP("ConvertLogicalOr");
    ConvertLogicalOr();
};

class ConvertLogicalXor: public ConvertLogicalBase {
public:
    OPENVINO_OP("ConvertLogicalXor");
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
