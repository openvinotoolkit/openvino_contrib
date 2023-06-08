// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda_eager_topology_runner.hpp"

namespace ov {
namespace nvidia_gpu {

class CudaGraphTopologyRunner final : public SubGraph, public ITopologyRunner {
public:
    struct CudaGraphIncompatible : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    CudaGraphTopologyRunner(const CreationContext& context, const std::shared_ptr<const ov::Model>& model);
    ~CudaGraphTopologyRunner() override = default;

    void Run(const InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const override;
    const SubGraph& GetSubGraph() const override;

    void Capture(InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const;
    void UpdateCapture(InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const;
};

}  // namespace nvidia_gpu
}  // namespace ov
