// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert.hpp"

#include <array>
#include <cuda/cuda_type_traits.hpp>
#include <cuda_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include <kernels/convert.hpp>
#include <utility>

#include "converters.hpp"

namespace ov {
namespace nvidia_gpu {

ConvertOp::ConvertOp(const CreationContext& context,
                     const std::shared_ptr<ov::Node>& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    Type_t input_element_type = node->get_input_element_type(0);
    Type_t output_element_type = node->get_output_element_type(0);
    OPENVINO_ASSERT(
        input_element_type >= Type_t::boolean && input_element_type <= Type_t::u64, "Node name: ", GetName());
    OPENVINO_ASSERT(
        output_element_type >= Type_t::boolean && output_element_type <= Type_t::u64, "Node name: ", GetName());
    if (input_element_type == Type_t::u1 || output_element_type == Type_t::u1)
        throw_ov_exception("Unsupported data type : Type_t::u1");
    auto input_shape = node->get_input_shape(0);
    auto output_shape = node->get_output_shape(0);
    const unsigned size = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());
    auto output_size = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<size_t>());
    OPENVINO_ASSERT(size == output_size, "Node name: ", GetName());
    const auto max_block_size = static_cast<unsigned>(context.device().props().maxThreadsPerBlock);
    const auto num_blocks = (size % max_block_size == 0) ? (size / max_block_size) : (size / max_block_size + 1);
    const auto threads_per_block = (num_blocks == 1) ? size : max_block_size;
    convert_kernel_ = kernel::Convert(convertDataType<ov::nvidia_gpu::kernel::Type_t>(output_element_type),
                                      convertDataType<ov::nvidia_gpu::kernel::Type_t>(input_element_type),
                                      size,
                                      num_blocks,
                                      threads_per_block);
}

void ConvertOp::Execute(const InferenceRequestContext& context,
                        Inputs inputs,
                        Outputs outputs,
                        const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() == 1, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());
    auto& threadContext = context.getThreadContext();
    auto& stream = threadContext.stream();
    (*convert_kernel_)(stream.get(), outputs[0].get(), inputs[0].get());
}

CudaGraphCompatibility ConvertOp::GetCudaGraphCompatibilityImpl() const { return CudaGraphCompatibility::FULL; }

OPERATION_REGISTER(ConvertOp, Convert);

}  // namespace nvidia_gpu
}  // namespace ov
