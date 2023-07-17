// Copyright (C) 2018-2021 Intel Corporation
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
    OPENVINO_ASSERT(context.has_input_tensor(input_tensor_name_), "Node name: ", GetName());
    auto tensor = context.get_input_tensor(input_tensor_name_);
    context.getThreadContext().stream().upload(outputs[0], tensor->data(), tensor->get_byte_size());
}

bool ParameterOp::IsCudaGraphCompatible() const { return true; }

std::string ParameterOp::GetInputTensorName(const ov::Node& node) { return node.get_friendly_name(); }

void ParameterOp::Capture(InferenceRequestContext &context, Inputs inputs, Outputs outputs,
                          const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() == 0, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());
    OPENVINO_ASSERT(context.has_input_tensor(input_tensor_name_), "Node name: ", GetName());
    auto tensor = context.get_input_tensor(input_tensor_name_);
    CUDA::CaptureInfo captureInfo{context.getThreadContext().stream()};
    context.getCudaGraphContext().parameterNodes.emplace(std::make_pair(input_tensor_name_,
            captureInfo.addUploadNode(outputs[0], tensor->data(), tensor->get_byte_size())));
}

OPERATION_REGISTER(ParameterOp, Parameter);

}  // namespace nvidia_gpu
}  // namespace ov
