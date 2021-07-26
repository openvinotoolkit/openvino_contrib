// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/variant.hpp>

namespace ArmPlugin {
namespace pass {

struct ConvertQuantize: public ngraph::pass::MatcherPass {
    ConvertQuantize();
};

struct NodeQuantizeFusion : public ngraph::pass::MatcherPass {
    NodeQuantizeFusion();
};

struct DequantizeNodeFusion : public ngraph::pass::FunctionPass{
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};

struct MovePerChenelQuantizationInfoToWeights : public ngraph::pass::MatcherPass {
    MovePerChenelQuantizationInfoToWeights();
};

struct PropogateQuantizationInfo: public ngraph::pass::FunctionPass {
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};

struct AddDequantizeOnInputs: public ngraph::pass::MatcherPass {
    AddDequantizeOnInputs();
};

struct ConvertBiasToI32: public ngraph::pass::MatcherPass {
    ConvertBiasToI32();
};

}  // namespace pass
}  // namespace ArmPlugin
