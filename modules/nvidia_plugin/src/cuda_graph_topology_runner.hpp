// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda_itopology_runner.hpp"

namespace ov {
namespace nvidia_gpu {

class CudaGraphTopologyRunner final : public ITopologyRunner {
public:
    CudaGraphTopologyRunner(const CreationContext& context, const std::shared_ptr<const ov::Model>& model);
    ~CudaGraphTopologyRunner() override = default;

    void Run(InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const override;
    void UpdateContext(InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const override;
    const SubGraph& GetSubGraph() const override;

    std::size_t GetCudaGraphsCount() const;

private:
    void Capture(InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const;
    void UpdateCapture(InferenceRequestContext& context) const;

    std::vector<SubGraph> subgraphs_;
    SubGraph orig_subgraph_;
    std::size_t cuda_graphs_count_;
};

}  // namespace nvidia_gpu
}  // namespace ov
