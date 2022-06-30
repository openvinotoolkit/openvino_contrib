// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_operation_registry.hpp>
#include "kernels/round.hpp"

#include "round.hpp"
#include "openvino/op/round.hpp"
#include "converters.hpp"

namespace CUDAPlugin {

RoundOp::RoundOp(const CreationContext& context,
                 const NodeOp& node,
                 IndexCollection&& inputIds,
                 IndexCollection&& outputIds)
    : OperationBase{context, node, move(inputIds), move(outputIds)} {
    Expects(node.get_input_size() == 1);
    Expects(node.get_output_size() == 1);

    const auto& element_type = node.get_input_element_type(0);
    const auto& out_element_type = node.get_output_element_type(0);
    if (out_element_type != element_type) {
        throwIEException(
            fmt::format("RoundOp: output type should be the same as input type, input type: {}, output type: {}",
                        element_type.get_type_name(),
                        out_element_type.get_type_name()));
    }
    const size_t num_elements = ov::shape_size(node.get_input_shape(0));
    Expects(ov::shape_size(node.get_output_shape(0)) == num_elements);

    const size_t max_threads_per_block = context.device().props().maxThreadsPerBlock;
    kernel_ = kernel::Round{
        convertDataType<CUDAPlugin::kernel::Type_t>(element_type), max_threads_per_block, num_elements};
}

void RoundOp::Execute(const InferenceRequestContext& context,
                      Inputs inputTensors,
                      Outputs outputTensors,
                      const Workbuffers& workbuffers) const {
    Expects(kernel_);
    Expects(inputTensors.size() == 1);
    Expects(outputTensors.size() == 1);

    (*kernel_)(context.getThreadContext().stream().get(), inputTensors[0].get(), outputTensors[0].get());
}

OPERATION_REGISTER(RoundOp, Round);

}
