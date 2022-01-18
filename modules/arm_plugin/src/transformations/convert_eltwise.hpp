// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "opset/opset.hpp"

namespace ArmPlugin {
namespace pass {

class ConvertEltwiseBase : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    template <class T>
    ngraph::matcher_pass_callback convert_eltwise();
};

class ConvertAdd: public ConvertEltwiseBase {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertAdd();
};

class ConvertSubtract: public ConvertEltwiseBase {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertSubtract();
};

class ConvertMultiply: public ConvertEltwiseBase {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertMultiply();
};

class ConvertMinimum: public ConvertEltwiseBase {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertMinimum();
};

class ConvertMaximum: public ConvertEltwiseBase {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertMaximum();
};

class ConvertEltwise: public ngraph::pass::GraphRewrite {
public:
    ConvertEltwise() {
        add_matcher<ConvertAdd>();
        add_matcher<ConvertSubtract>();
        add_matcher<ConvertMultiply>();
        add_matcher<ConvertMinimum>();
        add_matcher<ConvertMaximum>();
    }
};
}  // namespace pass
}  // namespace ArmPlugin
