// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "less.hpp"

#include <cuda_operation_registry.hpp>

namespace CUDAPlugin {

LessOp::LessOp(const CreationContext& context,
               const ngraph::Node& node,
               IndexCollection&& inputIds,
               IndexCollection&& outputIds)
    : Comparison(context, node, std::move(inputIds), std::move(outputIds), kernel::Comparison::Op_t::LESS) {}

OPERATION_REGISTER(LessOp, Less);

}  // namespace CUDAPlugin
