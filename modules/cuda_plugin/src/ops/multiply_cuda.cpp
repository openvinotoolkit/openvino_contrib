// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "multiply_cuda.hpp"

#include <cuda_operation_registry.hpp>
#include <ngraph/op/multiply.hpp>

#include "converters.hpp"

namespace CUDAPlugin {

MultiplyCudaOp::MultiplyCudaOp(const CreationContext& context,
                               const ngraph::Node& node,
                               IndexCollection&& inputIds,
                               IndexCollection&& outputIds)
    : OperationBase{context, node, move(inputIds), move(outputIds)} {
    auto multiplyOp = dynamic_cast<const ngraph::op::v1::Multiply*>(&node);
    Expects(multiplyOp);
    Expects(multiplyOp->get_input_size() == 2);
    Expects(multiplyOp->get_output_size() == 1);

    const auto element_type = multiplyOp->get_input_element_type(0);
    const bool types_expected = (element_type == multiplyOp->get_input_element_type(1)) &&
                                (element_type == multiplyOp->get_output_element_type(0));
    if (!types_expected) {
        throwIEException("MultiplyCuda: element types combination are not supported");
    }
    const auto& data_shape = multiplyOp->get_input_shape(0);
    const bool shapes_expected =
        (data_shape == multiplyOp->get_input_shape(1)) && (data_shape == multiplyOp->get_output_shape(0));
    if (!shapes_expected) {
        throwIEException("MultiplyCuda: shapes combination are not supported");
    }

    const size_t num_elements = std::accumulate(data_shape.begin(), data_shape.end(), 1, std::multiplies<size_t>());
    const size_t max_threads_per_block = context.device().props().maxThreadsPerBlock;

    kernel_ = kernel::Elementwise{kernel::Elementwise::Op_t::mul,
                                  convertDataType<CUDAPlugin::kernel::Type_t>(element_type),
                                  num_elements,
                                  max_threads_per_block};
}

void MultiplyCudaOp::Execute(const InferenceRequestContext& context,
                             Inputs inputTensors,
                             Outputs outputTensors,
                             const Workbuffers& workbuffers) const {
    Expects(kernel_);
    Expects(inputTensors.size() == 2);
    Expects(outputTensors.size() == 1);
    Expects(workbuffers.mutable_buffers.empty());
    auto& stream = context.getThreadContext().stream();
    (*kernel_)(stream.get(),
               static_cast<const void*>(inputTensors[0].get()),
               static_cast<const void*>(inputTensors[1].get()),
               static_cast<void*>(outputTensors[0].get()));
}

}  // namespace CUDAPlugin
