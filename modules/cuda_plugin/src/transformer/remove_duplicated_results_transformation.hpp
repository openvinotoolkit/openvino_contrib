// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph::pass {

class RemoveDuplicatedResultsTransformation : public ngraph::pass::FunctionPass {
 public:
    static constexpr auto Name = "RemoveDuplicatedResultsTransformation";

    NGRAPH_RTTI_DECLARATION;
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};

}  // namespace ngraph::pass
