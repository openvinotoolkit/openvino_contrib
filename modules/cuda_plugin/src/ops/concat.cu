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
template <typename T>
static __global__ void concat(const ConcatOp::Chunk* chunks,
                              const size_t allChunkSize,
                              const size_t numInputChunks,
                              const size_t chunkSize,
                              const T* const* src,
                              T* dst) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= allChunkSize) {
      return;
    }
    const unsigned chunkIdx = (i / chunkSize) % numInputChunks;
    const unsigned dataIdx = i % chunkSize;
    const auto& chunk = chunks[chunkIdx];
    dst[chunkIdx * chunkSize + dataIdx] = (src[chunk.input]+chunk.offset)[dataIdx];
}
} // namespace kernel

ConcatOp::ConcatOp(const CUDA::CreationContext& context,
                 const NodeOp& concatOp,
                 IndexCollection&& inputIds,
                 IndexCollection&& outputIds)
    : OperationBase(context, concatOp, std::move(inputIds), std::move(outputIds)),
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
    chunk_size_ = std::accumulate(outputShape.begin()+axis+1, outputShape.end(), 1, std::multiplies<size_t>());
    Ensures(chunk_size_ != 0);
    chunks_.reserve(num_chunks);
    const size_t sizeAboveAxis = num_chunks / outputShape[axis];
    for (size_t axis_above = 0; axis_above < sizeAboveAxis; axis_above++) {
        for (size_t curr_input = 0; curr_input < num_inputs_; curr_input++) {
            const auto& inputShape = concatOp.get_input_shape(curr_input);
            const size_t sizeAlongAxis = inputShape[axis];
            const size_t offset = axis_above * sizeAlongAxis;
            for (size_t pos_along_axis = 0; pos_along_axis < sizeAlongAxis; pos_along_axis++) {
                chunks_.emplace_back(Chunk{curr_input, (offset + pos_along_axis) * chunk_size_});
            }
        }
    }
    all_chunk_size_ = chunk_size_ * chunks_.size();
    const unsigned maxBlockSize = context.device().props().maxThreadsPerBlock;
    num_blocks_ = (all_chunk_size_ + maxBlockSize - 1) / maxBlockSize;
    threads_per_block_ = (num_blocks_ == 1) ? all_chunk_size_ : maxBlockSize;
}

WorkbufferRequest ConcatOp::GetWorkBufferRequest() const {
  return { {immutableWbSize()}, {mutableWbSize()} };
}

void ConcatOp::InitSharedImmutableWorkbuffers(const Buffers& buffers) {
  Expects(buffers.size() == 1);
  CUDA::DefaultStream::stream().upload(buffers[0], chunks_.data(), immutableWbSize());
}

void ConcatOp::Execute(const InferenceRequestContext& context, Inputs inputs, Outputs outputs, const Workbuffers& buffers) {
  switch (element_type_) {
    case ngraph::element::Type_t::boolean: return Execute<bool>(context, inputs, outputs, buffers);
    case ngraph::element::Type_t::bf16: return Execute<__nv_bfloat16>(context, inputs, outputs, buffers);
    case ngraph::element::Type_t::f16: return Execute<__half>(context, inputs, outputs, buffers);
    case ngraph::element::Type_t::f32: return Execute<float>(context, inputs, outputs, buffers);
    case ngraph::element::Type_t::f64: return Execute<double>(context, inputs, outputs, buffers);
    case ngraph::element::Type_t::i8: return Execute<int8_t>(context, inputs, outputs, buffers);
    case ngraph::element::Type_t::i16: return Execute<int16_t>(context, inputs, outputs, buffers);
    case ngraph::element::Type_t::i32: return Execute<int32_t>(context, inputs, outputs, buffers);
    case ngraph::element::Type_t::i64: return Execute<int64_t>(context, inputs, outputs, buffers);
    case ngraph::element::Type_t::u8: return Execute<uint8_t>(context, inputs, outputs, buffers);
    case ngraph::element::Type_t::u16: return Execute<uint16_t>(context, inputs, outputs, buffers);
    case ngraph::element::Type_t::u32: return Execute<uint32_t>(context, inputs, outputs, buffers);
    case ngraph::element::Type_t::u64: return Execute<uint64_t>(context, inputs, outputs, buffers);
    default: THROW_IE_EXCEPTION << fmt::format("Input element type = {} is not supported by Split operation !!",
                                               static_cast<ngraph::element::Type_t>(element_type_));
  }
}

template <typename T>
void ConcatOp::Execute(const InferenceRequestContext& context, Inputs inputs, Outputs outputs, const Workbuffers& workbuffers) {
    Expects(inputs.size() == num_inputs_);
    Expects(outputs.size() == 1);
    auto& threadContext = context.getThreadContext();
    auto& stream = threadContext.stream();

    Expects(all_chunk_size_);
    Expects(workbuffers.immutable_buffers.size()==1);
    Expects(workbuffers.mutable_buffers.size()==1);

    stream.upload(workbuffers.mutable_buffers[0], inputs.data(), mutableWbSize());
    kernel::concat<T><<<num_blocks_, threads_per_block_, 0, stream.get()>>>(
        reinterpret_cast<const Chunk*>(workbuffers.immutable_buffers[0].get()),
        all_chunk_size_,
        chunks_.size(),
        chunk_size_,
        reinterpret_cast<const T * const *>(workbuffers.mutable_buffers[0].get()),
        reinterpret_cast<T *>(outputs[0].get()));
}

OPERATION_REGISTER(ConcatOp, Concat);
} // namespace CUDAPlugin
