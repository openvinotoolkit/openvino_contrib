// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pass/pass.hpp>

namespace ngraph::pass {

class FullyConnectedTransformation : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    FullyConnectedTransformation();
};

}  // namespace ngraph::pass
