// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_sum.hpp"

#include <cuda/descriptor_utils.hpp>
#include <cuda_operation_registry.hpp>

namespace CUDAPlugin {

cudnnDataType_t reduceCompType(const ngraph::Node& node) {
    if (node.get_input_element_type(0) == ngraph::element::Type_t::f64) return CUDNN_DATA_DOUBLE;
    return CUDNN_DATA_FLOAT;  // TODO: it's unclear from documentation, whether it can be half when both tensors are
                              // half, or int8 when both tensors are int8. we'll have to test it
}

ReduceSumOp::ReduceSumOp(const CreationContext& context,
                         const ngraph::Node& node,
                         IndexCollection&& inputIds,
                         IndexCollection&& outputIds)
    : OperationCuDnn{context, node, move(inputIds), move(outputIds)},
      comp_type_{reduceCompType(node)},
      a_desc_{CUDA::makeInputDnnTensorDescr(node, 0)},
      c_desc_{CUDA::makeOutputDnnTensorDescr(node, 0)},
      workspace_size_{context.dnnHandle().getReductionWorkspaceSize(add_desc_, a_desc_, c_desc_)} {}

void ReduceSumOp::Execute(const InferenceRequestContext& context,
                          Inputs inputTensors,
                          Outputs outputTensors,
                          const Workbuffers& workbuffers) const {
    context.getThreadContext().dnnHandle().reduceTensor(add_desc_,
                                                        workbuffers.createMutableSpanFrom<0>(workspace_size_),
                                                        CUDA::DnnScaleFactorOne{comp_type_},
                                                        a_desc_,
                                                        inputTensors[0],
                                                        CUDA::DnnScaleFactorZero{comp_type_},
                                                        c_desc_,
                                                        outputTensors[0]);
}

OPERATION_REGISTER(ReduceSumOp, ReduceSum);
}  // namespace CUDAPlugin
