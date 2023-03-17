// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {

class ConvertCeiling: public ngraph::pass::MatcherPass {
public:
    OPENVINO_OP("ConvertCeiling");
    ConvertCeiling();
};
}  // namespace pass
}  // namespace ArmPlugin
