// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {

class ConvertBatchNormInferenceV0toV5;

} // namespace pass
} // namespace ArmPlugin

class ArmPlugin::pass::ConvertBatchNormInferenceV0toV5 : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertBatchNormInferenceV0toV5();
};
