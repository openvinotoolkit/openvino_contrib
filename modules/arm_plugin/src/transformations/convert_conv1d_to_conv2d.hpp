// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {
class ConvertConv1DBase: public ngraph::pass::MatcherPass {
protected:
    template <class Conv>
    ngraph::matcher_pass_callback convert_conv1d_to_conv2d();
};

class ConvertConv1D: public ConvertConv1DBase {
public:
    ConvertConv1D();
};

class ConvertGroupConv1D: public ConvertConv1DBase {
public:
    ConvertGroupConv1D();
};
}  // namespace pass
}  // namespace ArmPlugin
