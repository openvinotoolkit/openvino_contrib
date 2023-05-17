// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_sum.hpp"

#include <cuda/descriptor_utils.hpp>
#include <cuda_operation_registry.hpp>

#include "converters.hpp"

namespace ov {
namespace nvidia_gpu {

cudnnDataType_t reduceCompType(const ov::Node& node) {
    const auto in_type = convertDataType<cudnnDataType_t>(node.get_input_element_type(0));
    const auto out_type = convertDataType<cudnnDataType_t>(node.get_output_element_type(0));
    // if (node.get_input_element_type(0) == ov::element::Type_t::f64) return CUDNN_DATA_DOUBLE;
    switch (switchCase(in_type, out_type)) {
        case switchCase(CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT):
        case switchCase(CUDNN_DATA_FLOAT, CUDNN_DATA_HALF):
        case switchCase(CUDNN_DATA_FLOAT, CUDNN_DATA_INT8):
        case switchCase(CUDNN_DATA_HALF, CUDNN_DATA_FLOAT):
        case switchCase(CUDNN_DATA_INT8, CUDNN_DATA_FLOAT):
            // TODO: it's unclear from documentation, whether it can be half when both tensors are
            // half, or int8 when both tensors are int8. we'll have to test it
            return CUDNN_DATA_FLOAT;
        case switchCase(CUDNN_DATA_DOUBLE, CUDNN_DATA_DOUBLE):
            return CUDNN_DATA_DOUBLE;
        default:
            throw_ov_exception(fmt::format("ov::nvidia_gpu::reduceCompType(): Unsupported data types: in0 = {}, in1 = {}",
                                         toString(in_type),
                                         toString(out_type)));
    }
}

ReduceSumOp::ReduceSumOp(const CreationContext& context,
                         const ov::Node& node,
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

bool ReduceSumOp::IsCudaGraphCompatible() const { return true; }

OPERATION_REGISTER(ReduceSumOp, ReduceSum);
}  // namespace nvidia_gpu
}  // namespace ov
