// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ops/subgraph.hpp>

namespace ov {
namespace nvidia_gpu {

class ExecGraph final : public SubGraph {
public:
    ExecGraph(const CreationContext& context, const std::shared_ptr<const ov::Model>& model);
    ~ExecGraph() override = default;

    void Run(const InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const;
};

}  // namespace nvidia_gpu
}  // namespace ov
