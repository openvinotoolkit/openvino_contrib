// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_graph_topology_runner.hpp"
#include "cuda/graph.hpp"
#include "cuda/event.hpp"
#include "cuda_profiler.hpp"

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
        context.getProfiler().set_cuda_event_record_mode(CUDA::Event::RecordMode::External);
        Workbuffers workbuffers{};
        workbuffers.mutable_buffers.emplace_back(memoryBlock.view().data());
        SubGraph::Capture(context, {}, {}, workbuffers);
    }
    context.getCudaGraphContext().graphExec_.emplace(capture.getGraph());
}

const SubGraph& CudaGraphTopologyRunner::GetSubGraph() const {
    return *this;
}

void CudaGraphTopologyRunner::UpdateCapture(InferenceRequestContext &context,
                                            const DeviceMemBlock &memoryBlock) const {
    CudaGraphContext& graphContext = context.getCudaGraphContext();
    for (auto& pair : graphContext.parameterNodes)
        pair.second.update_src(graphContext.graphExec_.value(),
                const_cast<const void*>(context.get_input_tensor(pair.first)->data()));
    for (auto& pair : graphContext.resultNodes)
        pair.second.update_dst(graphContext.graphExec_.value(), context.get_output_tensor(pair.first)->data());
}

}  // namespace nvidia_gpu
}  // namespace ov
