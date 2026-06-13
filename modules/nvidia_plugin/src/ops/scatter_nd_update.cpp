// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "scatter_nd_update.hpp"

#include <fmt/format.h>

#include <cuda_operation_registry.hpp>
#include <openvino/op/scatter_nd_update.hpp>

#include "converters.hpp"

namespace ov {
namespace nvidia_gpu {

ScatterNDUpdateOp::ScatterNDUpdateOp(const CreationContext& context,
                                     const ov::Node& node,
                                     IndexCollection&& inputIds,
                                     IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    OPENVINO_ASSERT(node.get_input_size() == 3, "Node name: ", GetName());
    OPENVINO_ASSERT(node.get_output_size() == 1, "Node name: ", GetName());

    const ov::element::Type_t input_type = node.get_input_element_type(0);
    switch (input_type) {
        case ov::element::Type_t::dynamic:
        case ov::element::Type_t::u1:
            throw_ov_exception(fmt::format("Params element type = {} is not supported by ScatterNDUpdate operation!",
                                         static_cast<ov::element::Type_t>(input_type)));
    }

    // update type must be the same as the input type
    OPENVINO_ASSERT(node.get_input_element_type(2) == input_type, "Node name: ", GetName());
    // output type must be the same as the input type
    OPENVINO_ASSERT(node.get_output_element_type(0) == input_type, "Node name: ", GetName());

    const ov::element::Type_t indices_type = node.get_input_element_type(1);
    if (indices_type != ov::element::Type_t::i64 && indices_type != ov::element::Type_t::i32) {
        throw_ov_exception(
            fmt::format("Params index type = {} is not supported by ScatterNDUpdate operation!", indices_type));
    }

    const auto& input_shape = node.get_input_shape(0);
    const auto& output_shape = node.get_output_shape(0);
    OPENVINO_ASSERT(input_shape == output_shape, "Node name: ", GetName());

    const auto& indices_shape = node.get_input_shape(1);
    const auto& indices_last_dim = indices_shape.back();

    const size_t num_of_update_elements =
        std::accumulate(input_shape.begin() + indices_last_dim, input_shape.end(), 1, std::multiplies<size_t>());

    const size_t num_of_update_chunks =
        std::accumulate(indices_shape.cbegin(), indices_shape.cend() - 1, 1, std::multiplies<size_t>());

    const size_t num_of_input_elements =
        std::accumulate(input_shape.cbegin(), input_shape.cend(), 1, std::multiplies<size_t>());

    const auto max_block_size = context.device().props().maxThreadsPerBlock;

    // what is the most efficient way of using threads(per element or per chunk)?
    const bool thread_per_element = num_of_update_elements > num_of_update_chunks;
    const size_t num_of_items = thread_per_element ? num_of_update_elements : num_of_update_chunks;

    const size_t num_of_blocks{num_of_items % max_block_size == 0 ? num_of_items / max_block_size
                                                                  : num_of_items / max_block_size + 1};

    const size_t num_of_threads{num_of_blocks == 1 ? num_of_items : max_block_size};

    kernel_ = kernel::ScatterNDUpdate{convertDataType<ov::nvidia_gpu::kernel::Type_t>(input_type),
                                      convertDataType<ov::nvidia_gpu::kernel::Type_t>(indices_type),
                                      indices_last_dim,
                                      num_of_update_elements,
                                      num_of_input_elements,
                                      num_of_update_chunks,
                                      num_of_blocks,
                                      num_of_threads,
                                      thread_per_element};

    input_data_dim_pading_ = [&] {
        std::vector<size_t> padding(input_shape.size(), 1);
        for (size_t i{input_shape.size() - 1}; i > 0; --i) padding[i - 1] = padding[i] * input_shape[i];
        return padding;
    }();
}

void ScatterNDUpdateOp::Execute(const InferenceRequestContext& context,
                                Inputs inputs,
                                Outputs outputs,
                                const Workbuffers& workbuffers) const {
    OPENVINO_ASSERT(inputs.size() == 3, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());

    (*kernel_)(context.getThreadContext().stream().get(),
               inputs[0].get(),
               inputs[1].get(),
               inputs[2].get(),
               static_cast<const size_t*>(workbuffers.immutable_buffers[0].get()),
               outputs[0].get());
}

CudaGraphCompatibility ScatterNDUpdateOp::GetCudaGraphCompatibilityImpl() const { return CudaGraphCompatibility::FULL; }

template <typename T>
static auto size_in_bytes(const std::vector<T>& v) noexcept {
    return sizeof(T) * v.size();
}

WorkbufferRequest ScatterNDUpdateOp::GetWorkBufferRequest() const {
    return {{size_in_bytes(input_data_dim_pading_)}, {}};
}

void ScatterNDUpdateOp::InitSharedImmutableWorkbuffers(const Buffers& buffers) {
    auto& stream = CUDA::DefaultStream::stream();
    stream.upload(buffers[0], input_data_dim_pading_.data(), size_in_bytes(input_data_dim_pading_));
}

OPERATION_REGISTER(ScatterNDUpdateOp, ScatterNDUpdate);

}  // namespace nvidia_gpu
}  // namespace ov
