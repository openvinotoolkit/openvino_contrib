// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include "openvino/pass/graph_rewrite.hpp"

namespace ov::nvidia_gpu::pass {

class FuseConvolutionWithBiasAdd : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FuseConvolutionWithBiasAdd", "0");
    FuseConvolutionWithBiasAdd();
};

class FuseGroupConvolutionWithBiasAdd : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FuseGroupConvolutionWithBiasAdd", "0");
    FuseGroupConvolutionWithBiasAdd();
};

class FuseConvolutionWithBiasAddAdd : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FuseConvolutionWithBiasAddAdd", "0");
    FuseConvolutionWithBiasAddAdd();
};

class FuseGroupConvolutionWithBiasAddAdd : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FuseGroupConvolutionWithBiasAddAdd", "0");
    FuseGroupConvolutionWithBiasAddAdd();
};

class SinkReluToFusedConvolution : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SinkReluToFusedConvolution", "0");
    SinkReluToFusedConvolution();
};

class SinkSigmoidToFusedConvolution : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SinkSigmoidToFusedConvolution", "0");
    SinkSigmoidToFusedConvolution();
};

class SinkTanhToFusedConvolution : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SinkTanhToFusedConvolution", "0");
    SinkTanhToFusedConvolution();
};

class CudaFuseConvBiasAddActivation : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("CudaFuseConvBiasAddActivation", "0");
    CudaFuseConvBiasAddActivation();
};

class CudaFuseGroupConvBiasAddActivation : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("CudaFuseGroupConvBiasAddActivation", "0");
    CudaFuseGroupConvBiasAddActivation();
};

class CudaFuseConvBackpropDataAdd : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("CudaFuseConvBackpropDataAdd", "0");
    CudaFuseConvBackpropDataAdd();
};

}  // namespace ov::nvidia_gpu::pass
