// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda_operation_base.hpp"
#include "kernels/elu.hpp"
#include "openvino/op/elu.hpp"

namespace ov {
namespace nvidia_gpu {

class EluOp : public OperationBase {
public:
    EluOp(const CreationContext& context,
          const ov::Node& node,
          IndexCollection&& inputIds,
          IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibilityImpl() const override;

private:
    std::optional<kernel::Elu> kernel_;
};

}  // namespace nvidia_gpu
}  // namespace ov
