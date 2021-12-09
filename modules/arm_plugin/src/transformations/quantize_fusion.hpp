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

struct ConvolutionQuantizeFusion : public ngraph::pass::MatcherPass {
    ConvolutionQuantizeFusion();
};

struct MeanQuantizeFusion : public ngraph::pass::MatcherPass {
    MeanQuantizeFusion();
};

struct DequantizeInputFusion : public ngraph::pass::MatcherPass{
    DequantizeInputFusion();
};

struct AddDequantizeOnInputs: public ngraph::pass::MatcherPass {
    AddDequantizeOnInputs();
};

struct ConvertBiasToI32: public ngraph::pass::MatcherPass {
    ConvertBiasToI32();
};

}  // namespace pass
}  // namespace ArmPlugin
