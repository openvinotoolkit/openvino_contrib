// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_runtime.h>
#include <gsl/gsl_assert>
#include <ngraph/node.hpp>
#include <cuda_operation_registry.hpp>

#include "parameter.hpp"

namespace CUDAPlugin {

ParameterOp::ParameterOp(const CUDA::CreationContext& context,
                         const ngraph::Node& node,
                         IndexCollection&& inputIds,
                         IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    input_tensor_name_ = GetInputTensorName(node);
}

void ParameterOp::Execute(const InferenceRequestContext& context,
                          Inputs inputs,
                          Outputs outputs,
                          const Workbuffers&) const {
    Expects(inputs.size() == 0);
    Expects(outputs.size() == 1);
    Expects(context.HasInputBlob(input_tensor_name_));
    auto blob = context.GetInputBlob(input_tensor_name_);
    auto memory_ptr = blob->as<InferenceEngine::MemoryBlob>()->rmap();
    context.getThreadContext().stream().upload(outputs[0], memory_ptr, blob->byteSize());
}

std::string ParameterOp::GetInputTensorName(const ngraph::Node& node) {
    return node.get_friendly_name();
}

OPERATION_REGISTER(ParameterOp, Parameter);
} // namespace CUDAPlugin
