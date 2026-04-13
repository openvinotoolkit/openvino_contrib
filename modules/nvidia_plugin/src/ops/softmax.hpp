// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cudnn_ops_infer.h>

#include <cuda/device_pointers.hpp>
#include <cuda_operation_base.hpp>
#include <openvino/op/softmax.hpp>

namespace ov {
namespace nvidia_gpu {

class SoftmaxOp : public OperationCuDnn {
public:
    using NodeOp = ov::op::v1::Softmax;

    SoftmaxOp(const CreationContext& context,
              const NodeOp& node,
              IndexCollection&& inputIds,
              IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibilityImpl() const override;

private:
    void mapRankAxis(const ov::Shape& shape, int axis);
    std::array<int, 4> shape_;
    cudnnDataType_t type_;
    CUDA::DnnTensorDescriptor tensor_descriptor_;
};

}  // namespace nvidia_gpu
}  // namespace ov
