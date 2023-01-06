// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {

class ConvertBatchNormInferenceToARM;

} // namespace pass
} // namespace ArmPlugin

class ArmPlugin::pass::ConvertBatchNormInferenceToARM : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertBatchNormInferenceToARM();
};
