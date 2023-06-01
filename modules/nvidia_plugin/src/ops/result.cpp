// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "result.hpp"

#include <cuda_runtime.h>

#include <cuda_operation_registry.hpp>
#include <exec_graph_info.hpp>
#include <openvino/core/except.hpp>
#include <transformations/rt_info/fused_names_attribute.hpp>
#include <transformer/cuda_rt_info.hpp>
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
        if (context.has_output_tensor(outputName)) {
            tensor = context.get_output_tensor(outputName);
            break;
        }
    }
    OPENVINO_ASSERT(tensor != nullptr, "Node name: ", GetName());
    context.getThreadContext().stream().download(tensor->data(), inputs[0], tensor->get_byte_size());
}

std::optional<std::string> ResultOp::GetFusedOutputTensorName(const ov::Node::RTMap& rtInfo,
                                                              const std::string& resultName) {
    if (auto found = rtInfo.find(RtInfo::CUDA_FUSED_NAMES_MAPPING); found != rtInfo.end()) {
        const auto& original_names = found->second.as<std::string>();
        const auto foundPos = original_names.find("FUSED:");
        if (foundPos == 0) {
            auto original_names_mapping = original_names.substr(std::strlen("FUSED:"));
            auto foundNextMappingPos = original_names_mapping.find(';', 0);
            while (foundNextMappingPos != std::string::npos) {
                const auto& mapping = original_names_mapping.substr(0, foundNextMappingPos);
                const auto foundMappingPos = mapping.find('=');
                if (foundMappingPos == std::string::npos) {
                    break;
                }
                const auto& mappingResultName = mapping.substr(0, foundMappingPos);
                const auto& mappingOutputName = mapping.substr(foundMappingPos + 1);
                if (mappingResultName == resultName) {
                    return mappingOutputName;
                }
                original_names_mapping = original_names_mapping.substr(foundNextMappingPos + 1);
                foundNextMappingPos = original_names_mapping.find(';');
            }
        }
    }
    return std::nullopt;
}

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
    std::vector<std::string> outputNames;

    const auto& input = node.input_value(0);
    auto name = ov::op::util::get_ie_output_name(input);
    outputNames.push_back(name);

    auto resultName = node.get_friendly_name();
    const auto foundName = GetFusedOutputTensorName(input.get_node()->get_rt_info(), resultName);
    if (foundName) {
        outputNames.push_back(foundName.value());
    }

    // NOTE: New way of getting the fused names for OpenVINO 2.0 API
    // TODO: When support for old OpenVINO API will be stopped, consider using only this approach.
    //       Also see any issues with Tacatron2 network
    const auto& fusedResults = ov::getFusedNamesVector(input.get_node()->shared_from_this());
    outputNames.insert(outputNames.end(), fusedResults.begin(), fusedResults.end());

    return outputNames;
}

OPERATION_REGISTER(ResultOp, Result);
}  // namespace nvidia_gpu
}  // namespace ov
