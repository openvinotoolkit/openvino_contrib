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
    : orig_subgraph_{context, model} {
    if (!orig_subgraph_.IsCudaGraphCompatible())
        throw CudaGraphIncompatible{"The topology is incompatible with CUDA graphs."};

    std::vector<SubGraph::ExecSequence> sequences;
    SubGraph::ExecSequence currentSequence;
    const auto& origSequence = orig_subgraph_.getExecSequence();
    const auto totalSize = origSequence.size();
    if (totalSize == 0) {
        throw_ov_exception("ExecSequence size is 0");
    }
    bool isLastOpCompatible = origSequence[0]->IsCudaGraphCompatible();
    currentSequence.push_back(origSequence[0]);
    for (size_t i = 1; i < totalSize; ++i) {
        const auto& op = origSequence[i];
        if (op->IsCudaGraphCompatible() != isLastOpCompatible) {
            isLastOpCompatible = !isLastOpCompatible;
            sequences.emplace_back(std::move(currentSequence));
            currentSequence.clear();
        }
        currentSequence.push_back(op);
    }
    sequences.emplace_back(std::move(currentSequence));

    const auto& memoryManager = orig_subgraph_.memoryManager();
    for (auto&& sequence : sequences) {
        subgraphs_.emplace_back(context, model, std::move(sequence), memoryManager);
    }
}

void CudaGraphTopologyRunner::Run(const InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const {
    context.getCudaGraphContext().graphExec.value().launch(context.getThreadContext().stream());
}

void CudaGraphTopologyRunner::Capture(InferenceRequestContext &context,
                                      const DeviceMemBlock &memoryBlock) const {
    context.getProfiler().set_cuda_event_record_mode(CUDA::Event::RecordMode::External);
    const auto& stream = context.getThreadContext().stream();
    for (const auto& subgraph : subgraphs_) {
        if (subgraph.IsCudaGraphCompatible()) {
            CUDA::GraphCapture capture{stream};
            {
                auto scope = capture.getScope();
                Workbuffers workbuffers{};
                workbuffers.mutable_buffers.emplace_back(memoryBlock.view().data());
                subgraph.Capture(context, {}, {}, workbuffers);
            }
            const auto& graph = capture.getGraph();
            context.getCudaGraphContext().graph.emplace_back(graph);
            context.getCudaGraphContext().graphExec.emplace_back(graph);
        }
    }
}

const SubGraph& CudaGraphTopologyRunner::GetSubGraph() const {
    return orig_subgraph_;
}

void CudaGraphTopologyRunner::UpdateContext(InferenceRequestContext &context, const DeviceMemBlock &memoryBlock) const {
    if (context.getCudaGraphContext().graphExec)
        UpdateCapture(context);
    else
        Capture(context, memoryBlock);
}

void CudaGraphTopologyRunner::UpdateCapture(InferenceRequestContext &context) const {
    CudaGraphContext& graphContext = context.getCudaGraphContext();
    for (auto& pair : graphContext.parameterNodes)
        pair.second.updateSrc(graphContext.graphExec.value(),
                (context.get_input_tensor(pair.first)->data()));
    for (auto& pair : graphContext.resultNodes)
        pair.second.update_dst(graphContext.graphExec.value(), context.get_output_tensor(pair.first)->data());
}

}  // namespace nvidia_gpu
}  // namespace ov
