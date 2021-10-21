// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph::pass {

class FuseConvolutionWithBiasAdd : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    FuseConvolutionWithBiasAdd();
};

class FuseConvolutionWithBiasaddAdd : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    FuseConvolutionWithBiasaddAdd();
};

class SinkReluToFusedConvolution : public ngraph::pass::MatcherPass {
public:
    static constexpr auto Name = "SinkReluToFusedConvolution";

    NGRAPH_RTTI_DECLARATION;
    SinkReluToFusedConvolution();
};

class SinkSigmoidToFusedConvolution : public ngraph::pass::MatcherPass {
public:
    static constexpr auto Name = "SinkSigmoidToFusedConvolution";

    NGRAPH_RTTI_DECLARATION;
    SinkSigmoidToFusedConvolution();
};

class SinkTanhToFusedConvolution : public ngraph::pass::MatcherPass {
public:
    static constexpr auto Name = "SinkTanhToFusedConvolution";

    NGRAPH_RTTI_DECLARATION;
    SinkTanhToFusedConvolution();
};

class CudaFuseConvBiasAddActivation : public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    CudaFuseConvBiasAddActivation();
};

class CudaFuseConvBackpropDataAdd : public ngraph::pass::MatcherPass {
public:
    static constexpr auto Name = "FuseConvBackpropDataAdd";

    NGRAPH_RTTI_DECLARATION;
    CudaFuseConvBackpropDataAdd();
};

}  // namespace ngraph::pass
