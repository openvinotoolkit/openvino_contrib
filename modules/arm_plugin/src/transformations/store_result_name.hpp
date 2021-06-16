// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {
struct StoreResultName : public ngraph::pass::FunctionPass {
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};
}  // namespace pass
}  // namespace ArmPlugin
