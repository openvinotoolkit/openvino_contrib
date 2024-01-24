// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_eager_topology_runner.hpp"

namespace ov {
namespace nvidia_gpu {

EagerTopologyRunner::EagerTopologyRunner(const CreationContext& context, const std::shared_ptr<const ov::Model>& model)
    : SubGraph(context, model) {}

}  // namespace nvidia_gpu
}  // namespace ov
