// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda/descriptor_utils.hpp>
#include <cuda_operation_registry.hpp>

namespace CUDAPlugin {
namespace {

cudnnDataType_t reduceCompType(const ngraph::Node& node) {
    if (node.get_input_element_type(0) == ngraph::element::Type_t::f64) return CUDNN_DATA_DOUBLE;
    return CUDNN_DATA_FLOAT;  // TODO: it's unclear from documentation, whether it can be half when both tensors are
                              // half, or int8 when both tensors are int8. we'll have to test it
}

class ReduceSumOp : public OperationCuDnn {
    cudnnDataType_t compType;
    CUDA::DnnReduceAddDescriptor addDesc{compType};
    CUDA::DnnTensorDescriptor aDesc;
    CUDA::DnnTensorDescriptor cDesc;
    size_t workspaceSize;
    CUDA::DnnScalingFactorOne one{compType};
    CUDA::DnnScalingFactorZero zero{compType};

    WorkbufferRequest GetWorkBufferRequest() const override {
        return {{}, {workspaceSize}};  // TODO: find a way to allocate buffers from constructor
    }
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override {
        context.getThreadContext().dnnHandle().reduceTensor(addDesc,
                                                            workbuffers.mutableSpan<0>(workspaceSize),
                                                            one,
                                                            aDesc,
                                                            inputTensors[0],
                                                            zero,
                                                            cDesc,
                                                            outputTensors[0]);
    }

public:
    ReduceSumOp(const CreationContext& context,
                const ngraph::Node& node,
                IndexCollection&& inputIds,
                IndexCollection&& outputIds)
        : OperationCuDnn{context, node, move(inputIds), move(outputIds)},
          compType{reduceCompType(node)},
          aDesc{CUDA::makeInputDnnTensorDescr(node, 0)},
          cDesc{CUDA::makeOutputDnnTensorDescr(node, 0)},
          workspaceSize{context.dnnHandle().getReductionWorkspaceSize(addDesc, aDesc, cDesc)} {}
};

}  // namespace

OPERATION_REGISTER(ReduceSumOp, ReduceSum);

}  // namespace CUDAPlugin
