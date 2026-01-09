// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "parameter.hpp"

#include <cuda_runtime.h>

#include <cuda_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include <cuda_runtime_api.h>

namespace ov {
namespace nvidia_gpu {

ParameterOp::ParameterOp(const CreationContext& context,
                         const ov::Node& node,
                         IndexCollection&& inputIds,
                         IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    input_tensor_name_ = GetInputTensorName(node);
}

void ParameterOp::Execute(const InferenceRequestContext& context,
                          Inputs inputs,
                          Outputs outputs,
                          const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() == 0, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());
    OPENVINO_ASSERT(context.getTensorMappingContext().has_input_tensor(input_tensor_name_), "Node name: ", GetName());
    auto tensor = context.getTensorMappingContext().get_input_tensor(input_tensor_name_);
    context.getThreadContext().stream().upload(outputs[0], tensor->data(), tensor->get_byte_size());
}

CudaGraphCompatibility ParameterOp::GetCudaGraphCompatibility() const { return CudaGraphCompatibility::FULL; }

std::string ParameterOp::GetInputTensorName(const ov::Node& node) { return node.get_friendly_name(); }

void ParameterOp::Capture(InferenceRequestContext &context, Inputs inputs, Outputs outputs,
                          const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() == 0, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());
    OPENVINO_ASSERT(context.getTensorMappingContext().has_input_tensor(input_tensor_name_), "Node name: ", GetName());
    auto tensor = context.getTensorMappingContext().get_input_tensor(input_tensor_name_);
    context.getCudaGraphContext().add_parameter(
        input_tensor_name_, context.getThreadContext().stream(), outputs[0], tensor->data(), tensor->get_byte_size());
}

OPERATION_REGISTER(ParameterOp, Parameter);

}  // namespace nvidia_gpu
}  // namespace ov
