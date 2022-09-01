// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph::pass {

class NoopBroadcastTransformation : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    NoopBroadcastTransformation();
};

}  // namespace ngraph::pass
