// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph::pass {

class TransposeMatMulTransformation : public ngraph::pass::MatcherPass {
public:
    static constexpr auto Name = "TransposeMatMulTransformation";

    NGRAPH_RTTI_DECLARATION;
    TransposeMatMulTransformation();
};

}  // namespace ngraph::pass
