// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "result.hpp"

#include <cuda_runtime.h>

#include <cuda_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include <transformations/rt_info/fused_names_attribute.hpp>
#include <utility>

#include "nop_op.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace nvidia_gpu {

ResultOp::ResultOp(const CreationContext& context,
                   const NodeOp& node,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    output_tensor_names_ = GetOutputTensorName(node);
}

void ResultOp::Execute(const InferenceRequestContext& context,
                       Inputs inputs,
                       Outputs outputs,
                       const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() == 1, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 0, "Node name: ", GetName());
    std::shared_ptr<ov::Tensor> tensor;
    for (const auto& outputName : output_tensor_names_) {
        if (context.getTensorMappingContext().has_output_tensor(outputName)) {
            tensor = context.getTensorMappingContext().get_output_tensor(outputName);
            break;
        }
    }
    OPENVINO_ASSERT(tensor != nullptr, "Node name: ", GetName());
    context.getThreadContext().stream().download(tensor->data(), inputs[0], tensor->get_byte_size());
}

CudaGraphCompatibility ResultOp::GetCudaGraphCompatibility() const { return CudaGraphCompatibility::FULL; }

std::optional<std::size_t> ResultOp::GetOutputTensorSubIndex(const ov::Output<ov::Node>& node) {
    const auto& opRegistry = OperationRegistry::getInstance();
    const auto& opType = opRegistry.getOperationType(node.get_node()->shared_from_this());
    if (opType && std::type_index(typeid(NopOp)) == opType.value()) {
        for (const auto& in : node.get_node()->input_values()) {
            const auto& idx = GetOutputTensorSubIndex(in);
            if (idx) {
                return idx;
            }
        }
    } else if (node.get_node()->get_output_size() > 1) {
        return node.get_index();
    }

    return std::nullopt;
}

std::vector<std::string> ResultOp::GetOutputTensorName(const ov::op::v0::Result& node) {
    const auto& input = node.input_value(0);
    return ov::getFusedNamesVector(input.get_node()->shared_from_this());
}

void ResultOp::Capture(InferenceRequestContext& context,
                       Inputs inputs,
                       Outputs outputs,
                       const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() == 1, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 0, "Node name: ", GetName());
    std::shared_ptr<ov::Tensor> tensor;
    std::string outputTensorName{};
    for (const auto& outputName : output_tensor_names_) {
        if (context.getTensorMappingContext().has_output_tensor(outputName)) {
            tensor = context.getTensorMappingContext().get_output_tensor(outputName);
            outputTensorName = outputName;
            break;
        }
    }
    OPENVINO_ASSERT(tensor != nullptr, "Node name: ", GetName());
    context.getCudaGraphContext().add_result(
        outputTensorName, context.getThreadContext().stream(), tensor->data(), inputs[0], tensor->get_byte_size());
}

OPERATION_REGISTER(ResultOp, Result);
}  // namespace nvidia_gpu
}  // namespace ov
