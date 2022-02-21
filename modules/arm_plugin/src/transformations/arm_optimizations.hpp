// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {

class ArmOptimizations: public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ArmOptimizations(const bool lpt, const bool dump) : _lpt{lpt}, _dump{dump} {}
    bool run_on_function(std::shared_ptr<ov::Model> m) override;

    void Dump(const std::shared_ptr<ov::Model>& m, const std::string& postfix);

    bool _lpt = false;
    bool _dump = false;
};
}  // namespace pass
}  // namespace ArmPlugin
