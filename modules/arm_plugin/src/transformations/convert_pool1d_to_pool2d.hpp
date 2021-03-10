// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {
class ConvertPool1DBase: public ngraph::pass::MatcherPass {
protected:
    template <class Pool>
    ngraph::matcher_pass_callback convert_pool1d_to_pool2d();
};

class ConvertMaxPool1D: public ConvertPool1DBase {
public:
    ConvertMaxPool1D();
};

class ConvertAvgPool1D: public ConvertPool1DBase {
public:
    ConvertAvgPool1D();
};
}  // namespace pass
}  // namespace ArmPlugin
