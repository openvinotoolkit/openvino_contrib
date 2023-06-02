// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ops/subgraph.hpp>

namespace ov {
namespace nvidia_gpu {

struct ITopologyRunner {
    virtual void Run(const InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const = 0;
    virtual const SubGraph& GetSubGraph() const = 0;
    virtual ~ITopologyRunner() = default;
};

class ExecGraph final : public SubGraph, public ITopologyRunner {
public:
    ExecGraph(const CreationContext& context, const std::shared_ptr<const ov::Model>& model);
    ~ExecGraph() override = default;

    void Run(const InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const override;
    const SubGraph& GetSubGraph() const override;
};

}  // namespace nvidia_gpu
}  // namespace ov
