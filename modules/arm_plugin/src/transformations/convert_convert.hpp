// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {

class ConvertArmConvertBase : public ngraph::pass::MatcherPass {
public:
    OPENVINO_OP("ConvertArmConvertBase");
    template <class T>
    ngraph::matcher_pass_callback convert_to_arm_convert();
};

class ConvertArmConvert : public ConvertArmConvertBase {
public:
    OPENVINO_OP("ConvertArmConvert");
    ConvertArmConvert();
};

class ConvertArmConvertLike : public ConvertArmConvertBase {
public:
    OPENVINO_OP("ConvertArmConvertLike");
    ConvertArmConvertLike();
};
}  // namespace pass
}  // namespace ArmPlugin
