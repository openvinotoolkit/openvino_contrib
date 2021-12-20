// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <array>
#include <cuda/cuda_type_traits.hpp>
#include <cuda_operation_registry.hpp>
#include <gsl/gsl_assert>
#include <kernels/convert.hpp>
#include <ngraph/node.hpp>
#include <utility>

#include "convert.hpp"
#include "converters.hpp"

namespace CUDAPlugin {

ConvertOp::ConvertOp(const CreationContext& context,
                     const std::shared_ptr<ngraph::Node>& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    Type_t input_element_type = node->get_input_element_type(0);
    Type_t output_element_type = node->get_output_element_type(0);
    Expects(input_element_type >= Type_t::boolean && input_element_type <= Type_t::u64);
    Expects(output_element_type >= Type_t::boolean && output_element_type <= Type_t::u64);
    if (input_element_type == Type_t::u1 || output_element_type == Type_t::u1)
        throwIEException("Unsupported data type : Type_t::u1");
    auto input_shape = node->get_input_shape(0);
    auto output_shape = node->get_output_shape(0);
    const unsigned size = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());
    auto output_size = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<size_t>());
    Expects(size == output_size);
    const auto max_block_size = static_cast<unsigned>(context.device().props().maxThreadsPerBlock);
    const auto num_blocks = (size % max_block_size == 0) ? (size / max_block_size) : (size / max_block_size + 1);
    const auto threads_per_block = (num_blocks == 1) ? size : max_block_size;
    convert_kernel_ = kernel::Convert(convertDataType<CUDAPlugin::kernel::Type_t>(output_element_type),
                                      convertDataType<CUDAPlugin::kernel::Type_t>(input_element_type),
                                      size,
                                      num_blocks,
                                      threads_per_block);
}

void ConvertOp::Execute(const InferenceRequestContext& context,
                        Inputs inputs,
                        Outputs outputs,
                        const Workbuffers&) const {
    Expects(inputs.size() == 1);
    Expects(outputs.size() == 1);
    auto& threadContext = context.getThreadContext();
    auto& stream = threadContext.stream();
    (*convert_kernel_)(stream.get(), outputs[0].get(), inputs[0].get());
}

OPERATION_REGISTER(ConvertOp, Convert);

}  // namespace CUDAPlugin
