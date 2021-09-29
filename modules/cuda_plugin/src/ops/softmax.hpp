// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cudnn_ops_infer.h>

#include <cuda/device_pointers.hpp>
#include <cuda_operation_base.hpp>
#include <gpu/gpu_context_api_cuda.hpp>
#include <ngraph/op/softmax.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/type/element_type.hpp>

namespace CUDAPlugin {

class SoftmaxOp : public OperationCuDnn {
public:
    using NodeOp = ngraph::op::v1::Softmax;
    SoftmaxOp(const CreationContext& context,
              const NodeOp& node,
              IndexCollection&& inputIds,
              IndexCollection&& outputIds);
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

private:
    void mapRankAxis(const ngraph::Shape& shape, int axis);
    std::array<int, 4> shape_;
    cudnnDataType_t type_;
    CUDA::DnnTensorDescriptor tensor_descriptor_;
};

} // namespace CUDAPlugin
