// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {

class ConvertQuantize: public ngraph::pass::MatcherPass {
public:
    OPENVINO_OP("ConvertQuantize");
    ConvertQuantize();
};

class ConvolutionQuantizeFusion : public ngraph::pass::MatcherPass {
public:
    OPENVINO_OP("ConvolutionQuantizeFusion");
    ConvolutionQuantizeFusion();
};

class MeanQuantizeFusion : public ngraph::pass::MatcherPass {
public:
    OPENVINO_OP("MeanQuantizeFusion");
    MeanQuantizeFusion();
};

class DequantizeInputFusion : public ngraph::pass::MatcherPass{
public:
    OPENVINO_OP("DequantizeInputFusion");
    DequantizeInputFusion();
};

class AddDequantizeOnInputs: public ngraph::pass::MatcherPass {
public:
    OPENVINO_OP("AddDequantizeOnInputs");
    AddDequantizeOnInputs();
};

class ConvertBiasToI32: public ngraph::pass::MatcherPass {
public:
    OPENVINO_OP("ConvertBiasToI32");
    ConvertBiasToI32();
};

}  // namespace pass
}  // namespace ArmPlugin
