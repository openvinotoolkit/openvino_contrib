// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph::pass {

class FuseConvolutionWithBiasAdd : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    FuseConvolutionWithBiasAdd();
};

class FuseGroupConvolutionWithBiasAdd : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    FuseGroupConvolutionWithBiasAdd();
};

class FuseConvolutionWithBiasAddAdd : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    FuseConvolutionWithBiasAddAdd();
};

class FuseGroupConvolutionWithBiasAddAdd : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    FuseGroupConvolutionWithBiasAddAdd();
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

class CudaFuseGroupConvBiasAddActivation : public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    CudaFuseGroupConvBiasAddActivation();
};

class CudaFuseConvBackpropDataAdd : public ngraph::pass::MatcherPass {
public:
    static constexpr auto Name = "FuseConvBackpropDataAdd";

    NGRAPH_RTTI_DECLARATION;
    CudaFuseConvBackpropDataAdd();
};

}  // namespace ngraph::pass
