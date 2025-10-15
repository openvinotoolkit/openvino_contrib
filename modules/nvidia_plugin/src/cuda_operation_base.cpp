// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>

#include "cuda_operation_base.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace nvidia_gpu {

OperationBase::OperationBase(const CreationContext& /*context*/,
                             const ov::Node& node,
                             IndexCollection&& inputIds,
                             IndexCollection&& outputIds)
    : node_name_{node.get_friendly_name()},
      type_name_{node.get_type_info().name},
      input_ids_{inputIds},
      output_ids_{outputIds} {
    if (node.get_input_size() > 0) {
        runtime_precision_ = node.get_input_element_type(0);
    } else if (node.get_output_size() > 0) {
        runtime_precision_ = node.get_output_element_type(0);
    }
}

}  // namespace nvidia_gpu
}  // namespace ov
