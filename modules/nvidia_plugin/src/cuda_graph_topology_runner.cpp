// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_graph_topology_runner.hpp"

#include "cuda/event.hpp"
#include "ops/tensor_iterator.hpp"

namespace ov {
namespace nvidia_gpu {

CudaGraphTopologyRunner::CudaGraphTopologyRunner(const CreationContext& context, const SubGraph& subgraph)
    : orig_subgraph_(subgraph), cuda_graphs_count_{0} {
    std::vector<SubGraph::ExecSequence> sequences;
    SubGraph::ExecSequence currentSequence;
    const auto& origSequence = orig_subgraph_.getExecSequence();
    const auto totalSize = origSequence.size();
    OPENVINO_ASSERT(totalSize != 0, "ExecSequence size is 0");

    CudaGraphCompatibility lastOpCompatibility = origSequence[0]->GetCudaGraphCompatibility();
    currentSequence.push_back(origSequence[0]);
    for (std::size_t i = 1; i < totalSize; ++i) {
        const auto& op = origSequence[i];
        auto comp = op->GetCudaGraphCompatibility();
        if (comp != lastOpCompatibility || comp == CudaGraphCompatibility::SPECIAL) {
            lastOpCompatibility = comp;
            sequences.emplace_back(std::move(currentSequence));
            currentSequence.clear();
        }
        if (comp == CudaGraphCompatibility::SPECIAL) {
            auto sg = std::dynamic_pointer_cast<SubGraph>(op);
            sg->initializeRunner();
            cuda_graphs_count_ += sg->GetCudaGraphsCount();
        }
        currentSequence.push_back(op);
    }
    sequences.emplace_back(std::move(currentSequence));

    const auto& model = orig_subgraph_.getModel();
    const auto& memoryManager = orig_subgraph_.memoryManager();
    for (const auto& sequence : sequences) {
        subgraphs_.emplace_back(context, model, sequence, memoryManager);
        if (subgraphs_.back().GetCudaGraphCompatibility() == CudaGraphCompatibility::FULL) {
            ++cuda_graphs_count_;
        }
    }
}

CudaGraphTopologyRunner::CudaGraphTopologyRunner(const CreationContext& context,
                                                 const std::shared_ptr<const ov::Model>& model)
    : CudaGraphTopologyRunner(context, {context, model}) {}

CudaGraphTopologyRunner::CudaGraphTopologyRunner(const CreationContext& context,
                                                 const std::shared_ptr<const ov::Model>& model,
                                                 const SubGraph::ExecSequence& sequence,
                                                 const std::shared_ptr<MemoryManager>& memoryManager)
    : CudaGraphTopologyRunner(context, {context, model, sequence, memoryManager}) {}

void CudaGraphTopologyRunner::Run(InferenceRequestContext& context, const Workbuffers& workbuffers) const {
    const auto& stream = context.getThreadContext().stream();
    auto& graphPack = context.getCurrentCudaGraphInfo();
    std::size_t graphIndex = 0;
    for (auto& subgraph : subgraphs_) {
        auto compatibility = subgraph.GetCudaGraphCompatibility();
        if (compatibility == CudaGraphCompatibility::FULL) {
            graphPack.select_current_graph(graphIndex);
            graphPack.launch(stream);
            graphIndex++;
        } else if (compatibility == CudaGraphCompatibility::SPECIAL) {
            graphPack.select_current_graph(graphIndex);
            context.setCurrentCudaGraphInfo(graphPack.get_current_graph());
            subgraph.ExecuteGraph(context, {}, {}, workbuffers);
            graphIndex++;
        } else {
            subgraph.Execute(context, {}, {}, workbuffers);
        }
    }
}

void CudaGraphTopologyRunner::Run(InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const {
    Workbuffers workbuffers{};
    workbuffers.mutable_buffers.emplace_back(memoryBlock.view().data());
    context.setCurrentCudaGraphInfo(context.getCudaGraphContext());
    Run(context, workbuffers);
}

void CudaGraphTopologyRunner::Capture(InferenceRequestContext& context, const Workbuffers& workbuffers) const {
    const auto& stream = context.getThreadContext().stream();
    auto& graphPack = context.getCurrentCudaGraphInfo();
    graphPack.reset();
    for (const auto& subgraph : subgraphs_) {
        auto compatibility = subgraph.GetCudaGraphCompatibility();
        if (compatibility == CudaGraphCompatibility::FULL) {
            graphPack.add(CudaGraphInfo::create());
            CUDA::GraphCapture capture{stream};
            {
                auto scope = capture.getScope();
                subgraph.Capture(context, {}, {}, workbuffers);
            }
            graphPack.set_current_graph(capture.getGraph());
        } else if (compatibility == CudaGraphCompatibility::SPECIAL) {
            auto& currentGraph =
                hasNestedRunners() ? graphPack.add(CudaGraphContext::create()) : graphPack.add(CudaGraphInfo::create());
            context.setCurrentCudaGraphInfo(currentGraph);
            subgraph.Capture(context, {}, {}, workbuffers);
        }
    }
    OPENVINO_ASSERT(cuda_graphs_count_ == graphPack.get_graphs_count());
}

void CudaGraphTopologyRunner::Capture(InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const {
    Workbuffers workbuffers{};
    workbuffers.mutable_buffers.emplace_back(memoryBlock.view().data());
    context.setCurrentCudaGraphInfo(context.getCudaGraphContext());
    Capture(context, workbuffers);
}

const SubGraph& CudaGraphTopologyRunner::GetSubGraph() const {
    return orig_subgraph_;
}

std::size_t CudaGraphTopologyRunner::GetCudaGraphsCount() const { return cuda_graphs_count_; }

bool CudaGraphTopologyRunner::hasNestedRunners() const {
    return std::any_of(
        subgraphs_.begin(), subgraphs_.end(), [](const SubGraph& sg) { return sg.hasTopologyRunners(); });
}

void CudaGraphTopologyRunner::UpdateContext(InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const {
    if (context.getCudaGraphContext().is_initialized()) {
        UpdateCapture(context);
    } else {
        Capture(context, memoryBlock);
    }
}

void CudaGraphTopologyRunner::UpdateCapture(InferenceRequestContext& context) const {
    context.getCudaGraphContext().update_capture(context.getTensorMappingContext());
}

}  // namespace nvidia_gpu
}  // namespace ov
