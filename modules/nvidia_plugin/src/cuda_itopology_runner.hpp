// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ops/subgraph.hpp>

namespace ov {
namespace nvidia_gpu {

struct ITopologyRunner {
    virtual void Run(const InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const = 0;
    virtual void UpdateContext(InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const = 0;
    virtual const SubGraph& GetSubGraph() const = 0;
    virtual ~ITopologyRunner() = default;
};

}  // namespace nvidia_gpu
}  // namespace ov
