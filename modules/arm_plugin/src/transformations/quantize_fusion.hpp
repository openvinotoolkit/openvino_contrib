// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {

class ConvertQuantize: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertQuantize");
    ConvertQuantize();
};

class ConvolutionQuantizeFusion : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvolutionQuantizeFusion");
    ConvolutionQuantizeFusion();
};

class MeanQuantizeFusion : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("MeanQuantizeFusion");
    MeanQuantizeFusion();
};

class DequantizeInputFusion : public ngraph::pass::MatcherPass{
public:
    OPENVINO_RTTI("DequantizeInputFusion");
    DequantizeInputFusion();
};

class AddDequantizeOnInputs: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("AddDequantizeOnInputs");
    AddDequantizeOnInputs();
};

class ConvertBiasToI32: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertBiasToI32");
    ConvertBiasToI32();
};

}  // namespace pass
}  // namespace ArmPlugin
