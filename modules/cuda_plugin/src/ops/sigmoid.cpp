// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cuda_operation_base.hpp>
#include <cuda_operation_registry.hpp>
#include <cuda/descriptor_utils.hpp>

namespace CUDAPlugin {
namespace  {

class Sigmoid : public OperationCuDnn {
  CUDA::SigmoidDescriptor sigmoidDesc;
  CUDA::DnnTensorDescriptor xDesc;
  CUDA::DnnTensorDescriptor yDesc;
  static inline float one = 1;
  static inline float zero = 0;

 public:
  Sigmoid(const CUDA::CreationContext& context, const std::shared_ptr<ngraph::Node>& node,
          IndexCollection&& inputIds, IndexCollection&& outputIds)
      : OperationCuDnn{context, node, move(inputIds), move(outputIds)},
        xDesc{CUDA::makeInputDnnTensorDescr(*node, 0)},
        yDesc{CUDA::makeOutputDnnTensorDescr(*node, 0)} {}
  void Execute(const InferenceRequestContext& context,
               Inputs inputTensors,
               Outputs outputTensors,
               const Workbuffers&) const override {
      context.getThreadContext().dnnHandle().activationForward(
          sigmoidDesc, &one, xDesc, inputTensors[0].get(), &zero, yDesc, outputTensors[0].get());
  }
};


}  // namespace

OPERATION_REGISTER(Sigmoid, Sigmoid);
}  // namespace CUDAPlugin
