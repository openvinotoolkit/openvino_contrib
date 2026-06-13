// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cudnn_ops_infer.h>

#include <cuda/dnn.hpp>
#include <cuda_operation_base.hpp>
#include <initializer_list>

namespace ov {
namespace nvidia_gpu {

class ActivationForwardCuDnnOpBase : public OperationCuDnn {
public:
    static constexpr std::size_t max_shape_size = 5;

    static constexpr std::initializer_list<cudnnDataType_t> supported_types{
        CUDNN_DATA_FLOAT, CUDNN_DATA_DOUBLE, CUDNN_DATA_HALF, CUDNN_DATA_INT8};

    ActivationForwardCuDnnOpBase(std::unique_ptr<CUDA::DnnActivationDescriptor> opDesc,
                                 const CreationContext& context,
                                 const ov::Node& node,
                                 IndexCollection&& inputIds,
                                 IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers&) const override;

    CudaGraphCompatibility GetCudaGraphCompatibilityImpl() const override;

protected:
    std::unique_ptr<CUDA::DnnActivationDescriptor> op_desc_;
    CUDA::DnnTensorDescriptor x_desc_;
    CUDA::DnnTensorDescriptor y_desc_;
    cudnnDataType_t data_type_;
};

}  // namespace nvidia_gpu
}  // namespace ov
