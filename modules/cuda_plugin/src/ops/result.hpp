// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda/device_pointers.hpp>
#include <cuda_operation_base.hpp>
#include <ngraph/op/result.hpp>

namespace CUDAPlugin {

class ResultOp : public OperationBase {
public:
    using NodeOp = ngraph::op::v0::Result;
    ResultOp(const CreationContext& context,
             const NodeOp& node,
             IndexCollection&& inputIds,
             IndexCollection&& outputIds);
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;
    static std::vector<std::string> GetOutputTensorName(const ngraph::op::Result& node);

private:
    static std::optional<std::size_t> GetOutputTensorSubIndex(const ngraph::Output<ngraph::Node>& node);
    static std::optional<std::string> GetFusedOutputTensorName(const ngraph::Node::RTMap& rtInfo,
                                                               const std::string& resultName);

    std::vector<std::string> output_tensor_names_;
};

}  // namespace CUDAPlugin
