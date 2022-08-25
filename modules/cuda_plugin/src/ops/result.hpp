// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda/device_pointers.hpp>
#include <cuda_operation_base.hpp>
#include <openvino/op/result.hpp>

namespace CUDAPlugin {

class ResultOp : public OperationBase {
public:
    using NodeOp = ov::op::v0::Result;
    ResultOp(const CreationContext& context,
             const NodeOp& node,
             IndexCollection&& inputIds,
             IndexCollection&& outputIds);
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;
    static std::vector<std::string> GetOutputTensorName(const ov::op::v0::Result& node);

private:
    static std::optional<std::size_t> GetOutputTensorSubIndex(const ov::Output<ov::Node>& node);
    static std::optional<std::string> GetFusedOutputTensorName(const ov::Node::RTMap& rtInfo,
                                                               const std::string& resultName);

    std::vector<std::string> output_tensor_names_;
};

}  // namespace CUDAPlugin
