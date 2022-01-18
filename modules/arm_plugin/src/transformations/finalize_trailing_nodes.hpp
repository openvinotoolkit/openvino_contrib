// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// #include <ngraph/pass/graph_rewrite.hpp>
#include "ngraph/pass/pass.hpp"

namespace ArmPlugin {
namespace pass {

class  FinalizeTrailingNodes: public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};
}  // namespace pass
}  // namespace ArmPlugin
