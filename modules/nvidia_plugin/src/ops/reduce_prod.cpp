// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_operation_registry.hpp"
#include "reduce_prod.hpp"
#include "kernels/reduce_prod_int.cuh"

namespace ov {
namespace nvidia_gpu {

ReduceProdOp::ReduceProdOp(const CreationContext& context,
                         const ov::Node& node,
                         IndexCollection&& inputIds,
                         IndexCollection&& outputIds)
    : ReduceOp(context, node, move(inputIds), move(outputIds), CUDA::DnnReduceMulDescriptor(reduceCompType(node))) {}

ReduceProdIntOp::ReduceProdIntOp(const CreationContext& context,
                                 const ov::Node& node,
                                 IndexCollection&& inputIds,
                                 IndexCollection&& outputIds)
    : OperationBase{context, node, std::move(inputIds), std::move(outputIds)},
      num_elements_{ov::shape_size(node.get_input_shape(0))} {
    OPENVINO_ASSERT(node.get_input_element_type(0) == ov::element::i32,
                    "ReduceProdIntOp only supports i32, got: ", node.get_input_element_type(0));
    // The single-block kernel reduces ALL input elements into output[0] and ignores
    // the axes input, so it is only correct for a full reduction producing a single
    // element, and only for up to one block (1024 elements). Fail loudly rather than
    // silently corrupt if a graph ever feeds a partial-axis or larger reduction.
    OPENVINO_ASSERT(ov::shape_size(node.get_output_shape(0)) == 1,
                    "ReduceProdIntOp supports only full-axis reduction to a scalar; output shape: ",
                    node.get_output_shape(0));
    OPENVINO_ASSERT(num_elements_ <= 1024,
                    "ReduceProdIntOp supports at most 1024 elements, got: ",
                    num_elements_);
}

void ReduceProdIntOp::Execute(const InferenceRequestContext& context,
                              Inputs inputTensors,
                              Outputs outputTensors,
                              const Workbuffers&) const {
    kernel::reduce_prod_int32(context.getThreadContext().stream().get(),
                              static_cast<const int32_t*>(inputTensors[0].get()),
                              static_cast<int32_t*>(outputTensors[0].get()),
                              num_elements_);
}

static OperationBase::Ptr reduceProdFactory(
    const CreationContext& context,
    const std::shared_ptr<ov::Node>& node,
    OperationBase::IndexCollection&& inputs,
    OperationBase::IndexCollection&& outputs) {
    const auto data_type = node->get_input_element_type(0);
    if (data_type == ov::element::i32) {
        return std::make_shared<ReduceProdIntOp>(context, *node, std::move(inputs), std::move(outputs));
    }
    return std::make_shared<ReduceProdOp>(context, *node, std::move(inputs), std::move(outputs));
}

OPERATION_REGISTER_FACTORY(reduceProdFactory, ReduceProd);

}  // namespace nvidia_gpu
}  // namespace ov
