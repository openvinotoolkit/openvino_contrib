// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_eager_topology_runner.hpp"

namespace ov {
namespace nvidia_gpu {

EagerTopologyRunner::EagerTopologyRunner(const CreationContext& context, const std::shared_ptr<const ov::Model>& model)
    : SubGraph(context, model) {}

void EagerTopologyRunner::Run(const InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const {
    Workbuffers workbuffers{};
    workbuffers.mutable_buffers.emplace_back(memoryBlock.view().data());
    SubGraph::Execute(context, {}, {}, workbuffers);
}

const SubGraph& EagerTopologyRunner::GetSubGraph() const {
    return *this;
}

}  // namespace nvidia_gpu
}  // namespace ov
