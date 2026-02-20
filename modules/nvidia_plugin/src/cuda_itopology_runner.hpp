// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_inference_request_context.hpp>
#include <memory_manager/cuda_workbuffers.hpp>

namespace ov {
namespace nvidia_gpu {

class SubGraph;

struct ITopologyRunner {
    virtual ~ITopologyRunner() = default;

    virtual void Run(InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const = 0;
    virtual void Run(InferenceRequestContext& context, const Workbuffers& workbuffers) const = 0;

    virtual void Capture(InferenceRequestContext& context, const Workbuffers& workbuffers) const = 0;
    virtual void UpdateContext(InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const = 0;

    virtual const SubGraph& GetSubGraph() const = 0;
    virtual std::size_t GetCudaGraphsCount() const = 0;
};

}  // namespace nvidia_gpu
}  // namespace ov
