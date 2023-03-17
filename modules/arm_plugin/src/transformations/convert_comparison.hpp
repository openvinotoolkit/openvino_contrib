// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {

class ConvertComparisionBase : public ngraph::pass::MatcherPass {
public:
    OPENVINO_OP("ConvertComparisionBase");
    template <class T>
    ngraph::matcher_pass_callback convert_comparision();
};

class ConvertEqual: public ConvertComparisionBase {
public:
    OPENVINO_OP("ConvertEqual");
    ConvertEqual();
};

class ConvertNotEqual: public ConvertComparisionBase {
public:
    OPENVINO_OP("ConvertNotEqual");
    ConvertNotEqual();
};

class ConvertGreater: public ConvertComparisionBase {
public:
    OPENVINO_OP("ConvertGreater");
    ConvertGreater();
};

class ConvertGreaterEqual: public ConvertComparisionBase {
public:
    OPENVINO_OP("ConvertGreaterEqual");
    ConvertGreaterEqual();
};

class ConvertLess: public ConvertComparisionBase {
public:
    OPENVINO_OP("ConvertLess");
    ConvertLess();
};

class ConvertLessEqual: public ConvertComparisionBase {
public:
    OPENVINO_OP("ConvertLessEqual");
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
