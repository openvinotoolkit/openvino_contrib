// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cudnn_ops_infer.h>

#include <cuda/dnn.hpp>
#include <cuda_operation_base.hpp>
#include <ngraph/node.hpp>

namespace CUDAPlugin {

class ActivationForwardCuDnnOpBase : public OperationCuDnn {
public:
    static constexpr std::size_t max_shape_size = 5;

    ActivationForwardCuDnnOpBase(std::unique_ptr<CUDA::DnnActivationDescriptor> opDesc,
                                 const CreationContext& context,
                                 const ngraph::Node& node,
                                 IndexCollection&& inputIds,
                                 IndexCollection&& outputIds);
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers&) const override;

protected:
    std::unique_ptr<CUDA::DnnActivationDescriptor> op_desc_;
    CUDA::DnnTensorDescriptor x_desc_;
    CUDA::DnnTensorDescriptor y_desc_;
    cudnnDataType_t data_type_;
};

}  // namespace CUDAPlugin
