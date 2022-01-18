// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/variant.hpp>

namespace ArmPlugin {
namespace pass {

class ConvertQuantize: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertQuantize();
};

class ConvolutionQuantizeFusion : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvolutionQuantizeFusion();
};

class MeanQuantizeFusion : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    MeanQuantizeFusion();
};

class DequantizeInputFusion : public ngraph::pass::MatcherPass{
public:
    NGRAPH_RTTI_DECLARATION;
    DequantizeInputFusion();
};

class AddDequantizeOnInputs: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    AddDequantizeOnInputs();
};

class ConvertBiasToI32: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertBiasToI32();
};

}  // namespace pass
}  // namespace ArmPlugin
