// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {

class ArmOptimizations: public ov::pass::ModelPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ArmOptimizations(const bool lpt, const bool dump) : _lpt{lpt}, _dump{dump} {}
    bool run_on_model(const std::shared_ptr<ov::Model> &m) override;

    void Dump(const std::shared_ptr<ov::Model>& m, const std::string& postfix);

    bool _lpt = false;
    bool _dump = false;
};
}  // namespace pass
}  // namespace ArmPlugin
