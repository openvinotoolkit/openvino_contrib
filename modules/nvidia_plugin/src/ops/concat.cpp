// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "concat.hpp"

#include <fmt/format.h>

#include <cuda_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include <utility>
#include <vector>

#include "converters.hpp"

namespace ov {
namespace nvidia_gpu {

ConcatOp::ConcatOp(const CreationContext& context,
                   const NodeOp& concatOp,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds)
    : OperationBase(context, concatOp, std::move(inputIds), std::move(outputIds)),
      num_inputs_{concatOp.get_input_size()} {
    const ov::element::Type element_type{concatOp.get_input_element_type(0)};
    auto output_element_type = concatOp.get_output_element_type(0);
    OPENVINO_ASSERT(concatOp.get_output_size() == 1, "Node name: ", GetName());
    OPENVINO_ASSERT(element_type == output_element_type, "Node name: ", GetName());
    OPENVINO_ASSERT(num_inputs_ == GetInputIds().size(), "Node name: ", GetName());
    OPENVINO_ASSERT(GetOutputIds().size() == 1, "Node name: ", GetName());
    const auto& outputShape = concatOp.get_output_shape(0);
    int64_t axis = concatOp.get_axis();
    if (axis < 0) {
        axis += static_cast<int64_t>(concatOp.get_input_partial_shape(0).rank().get_length());
    }
    OPENVINO_ASSERT(axis >= 0 && axis < outputShape.size(), "Node name: ", GetName());
    auto num_chunks =
        std::accumulate(outputShape.begin(), outputShape.begin() + axis + 1, 1, std::multiplies<size_t>());
    OPENVINO_ASSERT(num_chunks != 0, "Node name: ", GetName());
    const std::size_t chunk_size =
        std::accumulate(outputShape.begin() + axis + 1, outputShape.end(), 1, std::multiplies<size_t>());
    OPENVINO_ASSERT(chunk_size != 0, "Node name: ", GetName());
    std::vector<kernel::Concat::Chunk> chunks;
    chunks.reserve(num_chunks);
    const size_t sizeAboveAxis = num_chunks / outputShape[axis];
    for (size_t axis_above = 0; axis_above < sizeAboveAxis; axis_above++) {
        for (size_t curr_input = 0; curr_input < num_inputs_; curr_input++) {
            const auto& inputShape = concatOp.get_input_shape(curr_input);
            const size_t sizeAlongAxis = inputShape[axis];
            const size_t offset = axis_above * sizeAlongAxis;
            for (size_t pos_along_axis = 0; pos_along_axis < sizeAlongAxis; pos_along_axis++) {
                chunks.emplace_back(kernel::Concat::Chunk{curr_input, (offset + pos_along_axis) * chunk_size});
            }
        }
    }

    const std::size_t allChunkSize = chunk_size * chunks.size();
    const unsigned maxBlockSize = context.device().props().maxThreadsPerBlock;
    const std::size_t numBlocks = (allChunkSize + maxBlockSize - 1) / maxBlockSize;
    const std::size_t threadsPerBlock = (numBlocks == 1) ? allChunkSize : maxBlockSize;

    concat_kernel_ = kernel::Concat{convertDataType<ov::nvidia_gpu::kernel::Type_t>(element_type),
                                    num_inputs_,
                                    std::move(chunks),
                                    chunk_size,
                                    allChunkSize,
                                    numBlocks,
                                    threadsPerBlock};
}

WorkbufferRequest ConcatOp::GetWorkBufferRequest() const { return {{immutableWbSize()}, {mutableWbSize()}}; }

void ConcatOp::InitSharedImmutableWorkbuffers(const Buffers& buffers) {
    OPENVINO_ASSERT(buffers.size() == 1, "Node name: ", GetName());
    CUDA::DefaultStream::stream().upload(buffers[0], concat_kernel_.value().immutableWbData(), immutableWbSize());
}

void ConcatOp::Execute(const InferenceRequestContext& context,
                       Inputs inputs,
                       Outputs outputs,
                       const Workbuffers& workbuffers) const {
    OPENVINO_ASSERT(concat_kernel_, "Node name: ", GetName());
    OPENVINO_ASSERT(inputs.size() == num_inputs_, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());
    auto& threadContext = context.getThreadContext();
    auto& stream = threadContext.stream();

    OPENVINO_ASSERT(workbuffers.immutable_buffers.size() == 1, "Node name: ", GetName());
    OPENVINO_ASSERT(workbuffers.mutable_buffers.size() == 1, "Node name: ", GetName());

    stream.upload(workbuffers.mutable_buffers[0], inputs.data(), mutableWbSize());
    (*concat_kernel_)(stream.get(),
                      workbuffers.immutable_buffers[0].get(),
                      reinterpret_cast<const void* const*>(workbuffers.mutable_buffers[0].get()),
                      outputs[0].get());
}

CudaGraphCompatibility ConcatOp::GetCudaGraphCompatibility() const { return CudaGraphCompatibility::NONE; }

OPERATION_REGISTER(ConcatOp, Concat);
}  // namespace nvidia_gpu
}  // namespace ov
