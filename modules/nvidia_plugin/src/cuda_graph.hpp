// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ops/subgraph.hpp>

class ExecNetworkTest;

namespace ov {
namespace nvidia_gpu {

class CudaGraph final : public SubGraph {
public:
    friend class ::ExecNetworkTest;

    CudaGraph(const CreationContext& context, const std::shared_ptr<const ngraph::Function>& function);
    ~CudaGraph() override = default;

    void Run(const InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const;
};

}  // namespace nvidia_gpu
}  // namespace ov
