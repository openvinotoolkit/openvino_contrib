// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {
class ConvertReduceMultiAxisBase : public ngraph::pass::MatcherPass {
public:
    template <class T>
    ngraph::matcher_pass_callback convert_reduce();
};

class ConvertReduceProd: public ConvertReduceMultiAxisBase {
public:
    ConvertReduceProd();
};

class ConvertReduceMin: public ConvertReduceMultiAxisBase {
public:
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
