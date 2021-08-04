// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph::pass {

class ConcatTransformation : public ngraph::pass::MatcherPass {
   public:
    NGRAPH_RTTI_DECLARATION;
    ConcatTransformation();
};

}
