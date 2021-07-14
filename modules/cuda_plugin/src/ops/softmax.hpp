// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cudnn_ops_infer.h>
#include <ngraph/type/element_type.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/op/softmax.hpp>
#include <cuda_operation_base.hpp>
#include <gpu/device_pointers.hpp>
#include <gpu/gpu_context_api_cuda.hpp>

namespace CUDAPlugin {

class SoftmaxOp : public OperationCuDnn {
 public:
  using InferenceRequestContext = InferenceEngine::gpu::InferenceRequestContext;
  using NodeOp = ngraph::op::v1::Softmax;
  SoftmaxOp(const CUDA::CreationContext& context,
            const NodeOp& node,
            IndexCollection&& inputIds,
            IndexCollection&& outputIds);
  void Execute(const InferenceRequestContext& context,
               Inputs inputTensors,
               Outputs outputTensors,
               const Workbuffers& workbuffers) override;
 private:
  void mapRankAxis(const ngraph::Shape& shape, int axis);
  std::array<int, 4> shape_;
  cudnnDataType_t type_;
  CUDA::DnnTensorDescriptor tensor_descriptor_;
};

} // namespace CUDAPlugin
