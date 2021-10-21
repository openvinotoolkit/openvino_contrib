// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "result.hpp"

#include <cuda_runtime.h>

#include <cuda_operation_registry.hpp>
#include <exec_graph_info.hpp>
#include <gsl/gsl_assert>
#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <transformer/cuda_rt_info.hpp>
#include <utility>

namespace CUDAPlugin {

ResultOp::ResultOp(const CreationContext& context,
                   const NodeOp& node,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    output_tensor_name_ = GetOutputTensorName(node);
}

void ResultOp::Execute(const InferenceRequestContext& context,
                       Inputs inputs,
                       Outputs outputs,
                       const Workbuffers&) const {
    Expects(inputs.size() == 1);
    Expects(outputs.size() == 0);
    Expects(context.HasOutputBlob(output_tensor_name_));
    auto blob = context.GetOutputBlob(output_tensor_name_);
    auto memory_ptr = blob->as<InferenceEngine::MemoryBlob>()->wmap();
    context.getThreadContext().stream().download(memory_ptr, inputs[0], blob->byteSize());
}

std::optional<std::string> ResultOp::GetFusedOutputTensorName(const ngraph::Node::RTMap& rtInfo,
                                                              const std::string& resultName) {
    if (auto found = rtInfo.find(RtInfo::CUDA_FUSED_NAMES_MAPPING); found != rtInfo.end()) {
        const auto& original_names = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(found->second)->get();
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

std::string ResultOp::GetOutputTensorName(const ngraph::Node& node) {
    auto previousOutput = node.get_input_source_output(0);
    auto resultName = node.get_friendly_name();
    const auto foundName = GetFusedOutputTensorName(previousOutput.get_node()->get_rt_info(), resultName);
    if (foundName) {
        return foundName.value();
    }

    auto outputName = previousOutput.get_node()->get_friendly_name();
    if (previousOutput.get_node()->get_output_size() > 1) {
        outputName += '.' + std::to_string(previousOutput.get_index());
    }
    return outputName;
}

OPERATION_REGISTER(ResultOp, Result);
}  // namespace CUDAPlugin
