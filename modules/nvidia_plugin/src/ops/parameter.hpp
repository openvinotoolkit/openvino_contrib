// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda/device_pointers.hpp>
#include <cuda_operation_base.hpp>

namespace ov {
namespace nvidia_gpu {

class ParameterOp : public OperationBase {
public:
    ParameterOp(const CreationContext& context,
                const ov::Node& node,
                IndexCollection&& inputIds,
                IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    void Capture(InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibilityImpl() const override;
    static std::string GetInputTensorName(const ov::Node& node);

private:
    std::string input_tensor_name_;
};

}  // namespace nvidia_gpu
}  // namespace ov
