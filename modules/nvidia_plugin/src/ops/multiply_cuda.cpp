// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multiply_cuda.hpp"

#include <fmt/ostream.h>

namespace ov {
namespace nvidia_gpu {

MultiplyCudaOp::MultiplyCudaOp(const CreationContext& context,
                               const NodeOp& node,
                               IndexCollection&& inputIds,
                               IndexCollection&& outputIds)
    : MultiplyCudaOpBase{context, node, std::move(inputIds), std::move(outputIds)} {
    const auto broatcast_type = node.get_autob().m_type;
    switch (broatcast_type) {
        case ov::op::AutoBroadcastType::NONE:
        case ov::op::AutoBroadcastType::NUMPY:
            break;
        default:
            throw_ov_exception(fmt::format("MultiplyCudaOp: unsupported broadcast type: {}", broatcast_type));
    }
}

}  // namespace nvidia_gpu
}  // namespace ov
