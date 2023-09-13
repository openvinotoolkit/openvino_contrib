// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ops/subgraph.hpp>

#include "cuda_itopology_runner.hpp"

namespace ov {
namespace nvidia_gpu {

class EagerTopologyRunner final : public SubGraph, public ITopologyRunner {
public:
    EagerTopologyRunner(const CreationContext& context, const std::shared_ptr<const ov::Model>& model);
    ~EagerTopologyRunner() override = default;

    void Run(const InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const override;
    void UpdateContext(InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const override{};
    const SubGraph& GetSubGraph() const override;
};

}  // namespace nvidia_gpu
}  // namespace ov
