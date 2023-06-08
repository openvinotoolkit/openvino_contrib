// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_graph_topology_runner.hpp"
#include "cuda/graph.hpp"

namespace ov {
namespace nvidia_gpu {

CudaGraphTopologyRunner::CudaGraphTopologyRunner(const CreationContext& context, const std::shared_ptr<const ov::Model>& model)
    : SubGraph(context, model) {
    if (!IsCudaGraphCompatible())
        throw CudaGraphIncompatible{"The topology is incompatible with CUDA graphs."};
}

void CudaGraphTopologyRunner::Run(const InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const {
    context.getCudaGraphContext().graphExec_.value().launch(context.getThreadContext().stream());
}

void CudaGraphTopologyRunner::Capture(InferenceRequestContext &context,
                                      const DeviceMemBlock &memoryBlock) const {
    CUDA::GraphCapture capture{context.getThreadContext().stream()};
    {
        auto scope = capture.getScope();
        Workbuffers workbuffers{};
        workbuffers.mutable_buffers.emplace_back(memoryBlock.view().data());
        SubGraph::Execute(context, {}, {}, workbuffers);
    }
    context.getCudaGraphContext().graphExec_.emplace(capture.getGraph());
}

const SubGraph& CudaGraphTopologyRunner::GetSubGraph() const {
    return *this;
}

void CudaGraphTopologyRunner::UpdateCapture(InferenceRequestContext &context,
                                            const DeviceMemBlock &memoryBlock) const {
    CUDA::GraphCapture capture{context.getThreadContext().stream()};
    {
        auto scope = capture.getScope();
        Workbuffers workbuffers{};
        workbuffers.mutable_buffers.emplace_back(memoryBlock.view().data());
        SubGraph::Execute(context, {}, {}, workbuffers);
    }
    context.getCudaGraphContext().graphExec_.value().update(capture.getGraph());
}

}  // namespace nvidia_gpu
}  // namespace ov
