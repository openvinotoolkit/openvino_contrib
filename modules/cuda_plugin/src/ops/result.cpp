// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_runtime.h>
#include <gsl/gsl_assert>
#include <ngraph/node.hpp>
#include <cuda_operation_registry.hpp>
#include <utility>

#include "result.hpp"

namespace CUDAPlugin {

std::string get_name_from(const ngraph::Node& node) {
    return node.inputs().size() == 0
           ? node.get_friendly_name()
           : node.input(0).get_source_output().get_node()->get_friendly_name();
}

ResultOp::ResultOp(const NodeOp& node,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds)
    : OperationBase(node, std::move(inputIds), std::move(outputIds)) {
    output_tensor_name_ = get_name_from(node);
}

void ResultOp::Execute(const InferenceRequestContext& context, Inputs inputs, Outputs outputs) {
  Expects(inputs.size() == 1);
  Expects(outputs.size() == 0);
  Expects(context.HasOutputBlob(output_tensor_name_));
  auto blob = context.GetOutputBlob(output_tensor_name_);
  auto memory_ptr = blob->as<InferenceEngine::MemoryBlob>()->wmap();
  context.getThreadContext().stream().download(memory_ptr, inputs[0], blob->byteSize());
}

OPERATION_REGISTER(ResultOp, Result);
} // namespace CUDAPlugin
