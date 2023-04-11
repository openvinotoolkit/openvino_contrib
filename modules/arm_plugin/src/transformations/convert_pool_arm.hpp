// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {

class ConvertArmMaxPoolV1: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertArmMaxPoolV1");
    ConvertArmMaxPoolV1();
};

class ConvertArmMaxPoolV8: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertArmMaxPoolV8");
    ConvertArmMaxPoolV8();
};

class ConvertArmAvgPool: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertArmAvgPool");
    ConvertArmAvgPool();
};

}  // namespace pass
}  // namespace ArmPlugin
