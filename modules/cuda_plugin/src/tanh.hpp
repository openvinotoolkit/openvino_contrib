// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

namespace CUDAPlugin {

class Tanh : public OperationCuDnn {
public:
    Tanh(const CUDA::CreationContext& context,
         const std::shared_ptr<ngraph::Node>& node,
         IndexCollection&& inputIds,
         IndexCollection&& outputIds);
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers&) override;

public:
    CUDA::TanhDescriptor tanh_desc_;
    CUDA::DnnTensorDescriptor x_desc_;
    CUDA::DnnTensorDescriptor y_desc_;
    cudnnDataType_t data_type_;
};

}  // namespace CUDAPlugin
