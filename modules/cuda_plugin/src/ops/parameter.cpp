// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_runtime.h>
#include <gsl/gsl_assert>
#include <ngraph/node.hpp>
#include <cuda_operation_registry.hpp>

#include "parameter.hpp"

namespace CUDAPlugin {

ParameterOp::ParameterOp(const std::shared_ptr<ngraph::Node>& node,
                         IndexCollection&& inputIds,
                         IndexCollection&& outputIds)
    : OperationBase(node, std::move(inputIds), std::move(outputIds)) {
    auto prevNode = node->output(0);
    input_tensor_name_ = prevNode.get_node()->get_friendly_name();
}

void ParameterOp::Execute(const InferenceRequestContext& context, Inputs inputs, Outputs outputs) {
  Expects(inputs.size() == 0);
  Expects(outputs.size() == 1);
  Expects(context.HasInputBlob(input_tensor_name_));
  auto blob = context.GetInputBlob(input_tensor_name_);
  auto memory_ptr = blob->as<InferenceEngine::MemoryBlob>()->rmap();
  context.getThreadContext().stream().upload(outputs[0], memory_ptr, blob->byteSize());
}

OPERATION_REGISTER(ParameterOp, Parameter);
} // namespace CUDAPlugin
