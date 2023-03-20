// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// #include <ngraph/pass/graph_rewrite.hpp>
#include "ngraph/pass/pass.hpp"

namespace ArmPlugin {
namespace pass {

class  FinalizeTrailingNodes: public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("FinalizeTrailingNodes");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};
}  // namespace pass
}  // namespace ArmPlugin
