// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::nvidia_gpu::pass {

class CudaFuseMarkUpNodesOrder : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("CudaFuseMarkUpNodesOrder", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

class CudaFuseCleanUpNodesOrder : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("CudaFuseCleanUpNodesOrder", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

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

class SinkActivationToFusedConvolution : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SinkActivationToFusedConvolution", "0");
    SinkActivationToFusedConvolution();
};

class CudaConvolutionFusion : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("CudaConvolutionFusion", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

class CudaFuseConvBackpropDataAdd : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("CudaFuseConvBackpropDataAdd", "0");
    CudaFuseConvBackpropDataAdd();
};

}  // namespace ov::nvidia_gpu::pass
