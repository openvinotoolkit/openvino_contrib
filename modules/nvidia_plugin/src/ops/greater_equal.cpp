// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "greater_equal.hpp"

#include <cuda_operation_registry.hpp>

namespace ov {
namespace nvidia_gpu {

GreaterEqualOp::GreaterEqualOp(const CreationContext& context,
                               const ov::Node& node,
                               IndexCollection&& inputIds,
                               IndexCollection&& outputIds)
    : Comparison(context, node, std::move(inputIds), std::move(outputIds), kernel::Comparison::Op_t::GREATER_EQUAL) {}

OPERATION_REGISTER(GreaterEqualOp, GreaterEqual);

}  // namespace nvidia_gpu
}  // namespace ov
