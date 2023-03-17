// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {

class DecomposeNormalizeL2Add: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("DecomposeNormalizeL2Add");
    DecomposeNormalizeL2Add();
};
}  // namespace pass
}  // namespace ArmPlugin
