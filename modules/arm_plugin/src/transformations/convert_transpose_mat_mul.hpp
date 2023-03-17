// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {

class ConvertTransposeMatMul: public ngraph::pass::MatcherPass {
public:
    OPENVINO_OP("ConvertTransposeMatMul");
    ConvertTransposeMatMul();
};
}  // namespace pass
}  // namespace ArmPlugin
