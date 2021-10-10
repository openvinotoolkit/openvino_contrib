// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_graph.hpp"

namespace CUDAPlugin {

CudaGraph::CudaGraph(const CreationContext& context, const std::shared_ptr<const ngraph::Function>& function)
    : SubGraph(context, function) {}

void CudaGraph::Run(const InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const {
    Workbuffers workbuffers{};
    workbuffers.mutable_buffers.emplace_back(memoryBlock.view().data());
    SubGraph::Execute(context, {}, {}, workbuffers);
}

}  // namespace CUDAPlugin
