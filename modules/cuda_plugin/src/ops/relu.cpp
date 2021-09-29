// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cuda/descriptor_utils.hpp>
#include <cuda_operation_base.hpp>
#include <cuda_operation_registry.hpp>

namespace CUDAPlugin {
namespace {

class ReluOp : public OperationCuDnn {
    CUDA::ReluDescriptor reluDesc;
    CUDA::DnnTensorDescriptor xDesc;
    CUDA::DnnTensorDescriptor yDesc;
    static inline float one = 1;
    static inline float zero = 0;

public:
    ReluOp(const CreationContext& context,
           const std::shared_ptr<ngraph::Node>& node,
           IndexCollection&& inputIds,
           IndexCollection&& outputIds)
        : OperationCuDnn{context, node, move(inputIds), move(outputIds)},
          xDesc{CUDA::makeInputDnnTensorDescr(*node, 0)},
          yDesc{CUDA::makeOutputDnnTensorDescr(*node, 0)} {}
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers&) const override {
        context.getThreadContext().dnnHandle().activationForward(
            reluDesc, &one, xDesc, inputTensors[0].get(), &zero, yDesc, outputTensors[0].get());
    }
};

}  // namespace

OPERATION_REGISTER(ReluOp, Relu);
}  // namespace CUDAPlugin
