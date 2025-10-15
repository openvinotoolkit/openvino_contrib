// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_creation_context.hpp>
#include <cuda_itopology_runner.hpp>
#include <ops/subgraph.hpp>

namespace ov {
namespace nvidia_gpu {

class CudaGraphTopologyRunner final : public ITopologyRunner {
public:
    CudaGraphTopologyRunner(const CreationContext& context, const std::shared_ptr<const ov::Model>& model);

    CudaGraphTopologyRunner(const CreationContext& context,
                            const std::shared_ptr<const ov::Model>& model,
                            const SubGraph::ExecSequence& sequence,
                            const std::shared_ptr<MemoryManager>& memoryManager);

    ~CudaGraphTopologyRunner() override = default;

    void Run(InferenceRequestContext& context, const Workbuffers& workbuffers) const override;
    void Run(InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const override;

    void Capture(InferenceRequestContext& context, const Workbuffers& workbuffers) const override;
    void UpdateContext(InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const override;

    const SubGraph& GetSubGraph() const override;
    std::size_t GetCudaGraphsCount() const override;

    bool hasNestedRunners() const;

private:
    explicit CudaGraphTopologyRunner(const CreationContext& context, const SubGraph& subgraph);

    void Capture(InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const;
    void UpdateCapture(InferenceRequestContext& context) const;

    std::vector<SubGraph> subgraphs_;
    SubGraph orig_subgraph_;
    std::size_t cuda_graphs_count_;
};

}  // namespace nvidia_gpu
}  // namespace ov
