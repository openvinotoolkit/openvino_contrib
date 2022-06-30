// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_operation_base.hpp"

#include <ngraph/node.hpp>
#include <utility>

namespace CUDAPlugin {

OperationBase::OperationBase(const CreationContext& /*context*/,
                             const ov::Node& node,
                             IndexCollection&& inputIds,
                             IndexCollection&& outputIds)
    : node_name_{node.get_name()},
      type_name_{node.get_type_info().name},
      input_ids_{move(inputIds)},
      output_ids_{move(outputIds)} {}

}  // namespace CUDAPlugin
