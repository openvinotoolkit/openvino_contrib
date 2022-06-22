// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "floor.hpp"

#include <cuda_operation_registry.hpp>
#include <openvino/op/constant.hpp>

#include "converters.hpp"

namespace CUDAPlugin {

FloorOp::FloorOp(const CreationContext& context,
                 const ov::Node& node,
                 IndexCollection&& inputIds,
                 IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    Expects(node.get_input_size() == 1);
    Expects(node.get_output_size() == 1);
    const auto input_element_type = node.get_input_element_type(0);
    const auto output_element_type = node.get_output_element_type(0);
    Expects(input_element_type == output_element_type);
    const auto input_shape = node.get_input_shape(0);
    const auto output_shape = node.get_output_shape(0);
    Expects(input_shape == output_shape);
    size_t num_elements = ov::shape_size(input_shape);
    const size_t max_threads_per_block = context.device().props().maxThreadsPerBlock;
    kernel_ = kernel::Floor{
        convertDataType<CUDAPlugin::kernel::Type_t>(input_element_type), max_threads_per_block, num_elements};
}

void FloorOp::Execute(const InferenceRequestContext& context,
                      Inputs inputTensors,
                      Outputs outputTensors,
                      const Workbuffers& workbuffers) const {
    Expects(kernel_);
    Expects(inputTensors.size() == 1);
    Expects(outputTensors.size() == 1);
    const auto& stream = context.getThreadContext().stream();
    (*kernel_)(stream.get(), inputTensors[0].get(), outputTensors[0].get());
}

OPERATION_REGISTER(FloorOp, Floor);

}  // namespace CUDAPlugin
