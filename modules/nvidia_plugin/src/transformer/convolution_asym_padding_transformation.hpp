// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::nvidia_gpu::pass {

class ConvolutionAsymPaddingTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvolutionAsymPaddingTransformation", "0");
    ConvolutionAsymPaddingTransformation();
};

class GroupConvolutionAsymPaddingTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GroupConvolutionAsymPaddingTransformation", "0");
    GroupConvolutionAsymPaddingTransformation();
};

class ConvolutionBackpropDataAsymPaddingTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvolutionBackpropDataAsymPaddingTransformation", "0");
    ConvolutionBackpropDataAsymPaddingTransformation();
};

class GroupConvolutionBackpropDataAsymPaddingTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GroupConvolutionBackpropDataAsymPaddingTransformation", "0");
    GroupConvolutionBackpropDataAsymPaddingTransformation();
};

class FusedConvBackpropDataAsymPaddingTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FusedConvBackpropDataAsymPaddingTransformation", "0");
    FusedConvBackpropDataAsymPaddingTransformation();
};

}  // namespace ov::nvidia_gpu::pass
