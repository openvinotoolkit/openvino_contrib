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
    : OperationBase{context, node, move(inputIds), move(outputIds)},
      num_elements_{ngraph::shape_size(node.get_input_shape(0))},
      min_{node.get_min()},
      max_{node.get_max()} {
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
    Expects(ngraph::shape_size(node.get_output_shape(0)) == num_elements_);

    const size_t max_threads_per_block = context.device().props().maxThreadsPerBlock;
    kernel_ = kernel::Clamp{convertDataType<CUDAPlugin::kernel::Type_t>(element_type), max_threads_per_block};
}

void ClampCudaOp::Execute(const InferenceRequestContext& context,
                          Inputs inputTensors,
                          Outputs outputTensors,
                          const Workbuffers& workbuffers) const {
    Expects(kernel_);
    Expects(inputTensors.size() == 1);
    Expects(outputTensors.size() == 1);

    (*kernel_)(context.getThreadContext().stream().get(),
               inputTensors[0].get(),
               num_elements_,
               outputTensors[0].get(),
               min_,
               max_);
}

}  // namespace CUDAPlugin
