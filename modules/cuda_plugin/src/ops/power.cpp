// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "power.hpp"

#include <cuda_operation_registry.hpp>
#include <ngraph/op/power.hpp>
#include <ngraph/op/util/elementwise_args.hpp>

#include "converters.hpp"

namespace CUDAPlugin {

PowerOp::PowerOp(const CreationContext& context,
                 const ngraph::Node& node,
                 IndexCollection&& inputIds,
                 IndexCollection&& outputIds)
    : OperationBase{context, node, move(inputIds), move(outputIds)} {
    auto powerOp = dynamic_cast<const ngraph::op::v1::Power*>(&node);
    Expects(powerOp);
    Expects(powerOp->get_input_size() == 2);
    Expects(powerOp->get_output_size() == 1);

    const auto element_type = powerOp->get_input_element_type(0);
    const bool types_expected =
        (element_type == powerOp->get_input_element_type(1)) && (element_type == powerOp->get_output_element_type(0));
    if (!types_expected) {
        throwIEException("Power CUDA operation: element types combination are not supported");
    }
    in0_num_elements_ = ngraph::shape_size(powerOp->get_input_shape(0));
    in1_num_elements_ = ngraph::shape_size(powerOp->get_input_shape(1));
    if (powerOp->get_autob() == ngraph::op::AutoBroadcastType::NONE) {
        Expects(in0_num_elements_ == in1_num_elements_);
    }
    ngraph::PartialShape pshape = powerOp->get_input_partial_shape(0);
    Expects(
        ngraph::PartialShape::broadcast_merge_into(pshape, powerOp->get_input_partial_shape(1), powerOp->get_autob()));
    Expects(ngraph::PartialShape::merge_into(pshape, powerOp->get_output_shape(0)));

    const size_t max_threads_per_block = context.device().props().maxThreadsPerBlock;
    kernel_ = kernel::Power{convertDataType<CUDAPlugin::kernel::Type_t>(element_type), max_threads_per_block};
}

void PowerOp::Execute(const InferenceRequestContext& context,
                      Inputs inputTensors,
                      Outputs outputTensors,
                      const Workbuffers& workbuffers) const {
    Expects(kernel_);
    Expects(inputTensors.size() == 2);
    Expects(outputTensors.size() == 1);
    auto& stream = context.getThreadContext().stream();
    (*kernel_)(stream.get(),
               static_cast<const void*>(inputTensors[0].get()),
               in0_num_elements_,
               static_cast<const void*>(inputTensors[1].get()),
               in1_num_elements_,
               static_cast<void*>(outputTensors[0].get()));
}

OPERATION_REGISTER(PowerOp, Power)

}  // namespace CUDAPlugin
