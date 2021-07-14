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

ResultOp::ResultOp(const CUDA::CreationContext& context,
                   const NodeOp& node,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    output_tensor_name_ = GetOutputTensorName(node);
}

void ResultOp::Execute(const InferenceRequestContext& context, Inputs inputs, Outputs outputs, const Workbuffers&) {
  Expects(inputs.size() == 1);
  Expects(outputs.size() == 0);
  Expects(context.HasOutputBlob(output_tensor_name_));
  auto blob = context.GetOutputBlob(output_tensor_name_);
  auto memory_ptr = blob->as<InferenceEngine::MemoryBlob>()->wmap();
  context.getThreadContext().stream().download(memory_ptr, inputs[0], blob->byteSize());
}

std::string ResultOp::GetOutputTensorName(const ngraph::Node& node) {
    auto previousOutput = node.get_input_source_output(0);
    auto outputName = previousOutput.get_node()->get_friendly_name();
    if (previousOutput.get_node()->get_output_size() > 1) {
        outputName += '.' + std::to_string(previousOutput.get_index());
    }
    return outputName;
}

OPERATION_REGISTER(ResultOp, Result);
} // namespace CUDAPlugin
