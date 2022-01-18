// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {

class ConvertConv1DBase: public ngraph::pass::MatcherPass {
protected:
    NGRAPH_RTTI_DECLARATION;
    template <class Conv>
    ngraph::matcher_pass_callback convert_conv1d_to_conv2d();
};

class ConvertConv1D: public ConvertConv1DBase {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertConv1D();
};

class ConvertGroupConv1D: public ConvertConv1DBase {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertGroupConv1D();
};
}  // namespace pass
}  // namespace ArmPlugin
