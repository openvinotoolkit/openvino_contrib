// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {
class ArmOptimizations: public ngraph::pass::FunctionPass {
public:
    ArmOptimizations(const bool lpt, const bool dump) : _lpt{lpt}, _dump{dump} {}
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
    bool _lpt = false;
    bool _dump = false;
};
}  // namespace pass
}  // namespace ArmPlugin
