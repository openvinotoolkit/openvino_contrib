// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph::pass {

class ConvolutionAsymPaddingTransformation : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvolutionAsymPaddingTransformation();
};

class GroupConvolutionAsymPaddingTransformation : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    GroupConvolutionAsymPaddingTransformation();
};

class ConvolutionBackpropDataAsymPaddingTransformation : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvolutionBackpropDataAsymPaddingTransformation();
};

class GroupConvolutionBackpropDataAsymPaddingTransformation : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    GroupConvolutionBackpropDataAsymPaddingTransformation();
};

class FusedConvBackpropDataAsymPaddingTransformation : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    FusedConvBackpropDataAsymPaddingTransformation();
};

}  // namespace ngraph::pass
