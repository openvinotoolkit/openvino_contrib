// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "clamp_cuda.hpp"

#include <fmt/format.h>

#include <cuda_operation_registry.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/type/element_type.hpp>

#include "converters.hpp"
#include "error.hpp"

namespace CUDAPlugin {

ClampCudaOp::ClampCudaOp(const CreationContext& context,
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
            fmt::format("ClampCudaOp: output type should be the same as input type, input type: {}, output type: {}",
                        element_type.get_type_name(),
                        out_element_type.get_type_name()));
    }
    const size_t num_elements = ngraph::shape_size(node.get_input_shape(0));
    Expects(ngraph::shape_size(node.get_output_shape(0)) == num_elements);

    const size_t max_threads_per_block = context.device().props().maxThreadsPerBlock;
    const double min = node.get_min();
    const double max = node.get_max();
    kernel_ = kernel::Clamp{
        convertDataType<CUDAPlugin::kernel::Type_t>(element_type), max_threads_per_block, num_elements, min, max};
}

void ClampCudaOp::Execute(const InferenceRequestContext& context,
                          Inputs inputTensors,
                          Outputs outputTensors,
                          const Workbuffers& workbuffers) const {
    Expects(kernel_);
    Expects(inputTensors.size() == 1);
    Expects(outputTensors.size() == 1);

    (*kernel_)(context.getThreadContext().stream().get(), inputTensors[0].get(), outputTensors[0].get());
}

}  // namespace CUDAPlugin
