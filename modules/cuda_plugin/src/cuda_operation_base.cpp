// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>

#include <ngraph/node.hpp>

#include "cuda_operation_base.hpp"

namespace CUDAPlugin {

OperationBase::OperationBase(const std::shared_ptr<ngraph::Node>& node,
                             std::vector<unsigned> inputIds,
                             std::vector<unsigned> outputIds)
    : node_name_{node->get_friendly_name()}
    , input_ids_{std::move(inputIds)}
    , output_ids_{std::move(outputIds)} {
}

} // namespace CUDAPlugin
