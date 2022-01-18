// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {

class ConvertReduceMultiAxisBase : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    template <class T>
    ngraph::matcher_pass_callback convert_reduce();
};

class ConvertReduceProd: public ConvertReduceMultiAxisBase {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertReduceProd();
};

class ConvertReduceMin: public ConvertReduceMultiAxisBase {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertReduceMin();
};

class ConvertReduceMultiAxis: public ngraph::pass::GraphRewrite {
public:
    ConvertReduceMultiAxis() {
        add_matcher<ConvertReduceProd>();
        add_matcher<ConvertReduceMin>();
    }
};
}  // namespace pass
}  // namespace ArmPlugin
