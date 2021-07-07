// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <cuda_runtime.h>
#include <gsl/gsl_assert>
#include <cuda/device.hpp>
#include <cuda_operation_registry.hpp>
#include <utility>
#include <fmt/format.h>

#include "cuda/runtime.hpp"
#include "details/cuda_ngraph_import.hpp"
#include "concat.hpp"

namespace CUDAPlugin {
namespace kernel {
static __global__ void concat(const ConcatOp::Chunk* chunks, const size_t numInputChunks, const size_t chunkSize,
                              const void * const *src,  void * dst) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numInputChunks) {
      const auto& chunk = chunks[i];
      memcpy(reinterpret_cast<char*>(dst) + i*chunkSize, reinterpret_cast<const char*>(src[chunk.input])+chunk.offset, chunkSize);
    }
}
} // namespace kernel

ConcatOp::ConcatOp(const CUDA::Device& device,
                 const NodeOp& concatOp,
                 IndexCollection&& inputIds,
                 IndexCollection&& outputIds)
    : OperationBase(device, concatOp, std::move(inputIds), std::move(outputIds)),
      element_type_ {concatOp.get_input_element_type(0)},
      num_inputs_ {concatOp.get_input_size()} {
    auto output_element_type = concatOp.get_output_element_type(0);
    auto output_element_size = output_element_type.size();
    Expects(concatOp.get_output_size() == 1);
    Expects(element_type_ == output_element_type);
    Expects(num_inputs_ == GetInputIds().size());
    Expects(GetOutputIds().size() == 1);
    const auto& outputShape = concatOp.get_output_shape(0);
    const int64_t axis = concatOp.get_axis();
    Expects(axis >= 0 && axis < outputShape.size());
    auto num_chunks = std::accumulate(outputShape.begin(), outputShape.begin()+axis+1, 1, std::multiplies<size_t>());
    Expects(num_chunks != 0);
    chunk_size_ = output_element_size * std::accumulate(outputShape.begin()+axis+1, outputShape.end(), 1, std::multiplies<size_t>());
    Ensures(chunk_size_ != 0);
    chunks_.reserve(num_chunks);
    const size_t sizeAboveAxis = num_chunks / outputShape[axis];
    for (size_t axis_above = 0; axis_above < sizeAboveAxis; axis_above++) {
      for (size_t curr_input = 0; curr_input < num_inputs_; curr_input++) {
        const auto& inputShape = concatOp.get_input_shape(curr_input);
        const size_t sizeAlongAxis = inputShape[axis];
        const size_t offset = axis_above * sizeAlongAxis;
        for (size_t pos_along_axis = 0; pos_along_axis < sizeAlongAxis; pos_along_axis++)
          chunks_.emplace_back(Chunk{curr_input, (offset + pos_along_axis)*chunk_size_});
      }
    }
    Ensures(chunks_.size() == num_chunks);
}

WorkbufferRequest ConcatOp::GetWorkBufferRequest() const {
  return { {immutableWbSize()}, {mutableWbSize()} };
}

void ConcatOp::InitSharedImmutableWorkbuffers(const Buffers& buffers) {
  Expects(buffers.size() == 1);
  CUDA::DefaultStream::stream().upload(buffers[0], chunks_.data(), immutableWbSize());  
}

void ConcatOp::Execute(const InferenceRequestContext& context, Inputs inputs, Outputs outputs, const Workbuffers& workbuffers) {
    Expects(inputs.size() == num_inputs_);
    Expects(outputs.size() == 1);
    Expects(workbuffers.immutable_buffers.size()==1);
    Expects(workbuffers.mutable_buffers.size()==1);
    auto& threadContext = context.getThreadContext();
    auto& stream = threadContext.stream();
    const unsigned maxBlockSize = threadContext.device().props().maxThreadsPerBlock;
    const unsigned numBlocks = (chunks_.size() + maxBlockSize - 1) / maxBlockSize;
    const unsigned threadsPerBlock = (numBlocks == 1) ? chunks_.size() : maxBlockSize;
    stream.upload(workbuffers.mutable_buffers[0], inputs.data(), mutableWbSize());
    kernel::concat<<<numBlocks, threadsPerBlock, 0, stream.get()>>>(
        reinterpret_cast<const Chunk*>(workbuffers.immutable_buffers[0].get()),
        chunks_.size(),
        chunk_size_,
        reinterpret_cast<const void * const *>(workbuffers.mutable_buffers[0].get()),
        outputs[0].get());
}

OPERATION_REGISTER(ConcatOp, Concat);
} // namespace CUDAPlugin
