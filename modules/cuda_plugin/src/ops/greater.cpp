// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "greater.hpp"

#include <cuda_operation_registry.hpp>

namespace CUDAPlugin {

GreaterOp::GreaterOp(const CreationContext& context,
                     const ngraph::Node& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds)
    : Comparison(context, node, std::move(inputIds), std::move(outputIds), kernel::Comparison::Op_t::GREATER) {}

OPERATION_REGISTER(GreaterOp, Greater);

}  // namespace CUDAPlugin
