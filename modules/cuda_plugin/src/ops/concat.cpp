// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "details/cuda_ie_api_import_fix.hpp"
// ^^ must come before any other ie includes which use
// INFERENCE_ENGINE_DEPRECATED
#include "details/cuda_ngraph_import_fix.hpp"
// ^^ must come before any other ngraph includes which use
// NGRAPH_DEPRECATED
#include <fmt/format.h>

#include <cuda_operation_registry.hpp>
#include <gsl/gsl_assert>
#include <utility>
#include <vector>

#include "concat.hpp"
#include "converters.hpp"

namespace CUDAPlugin {

ConcatOp::ConcatOp(const CreationContext& context,
                   const NodeOp& concatOp,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds)
    : OperationBase(context, concatOp, std::move(inputIds), std::move(outputIds)),
      num_inputs_{concatOp.get_input_size()} {
    const ngraph::element::Type element_type{concatOp.get_input_element_type(0)};
    auto output_element_type = concatOp.get_output_element_type(0);
    Expects(concatOp.get_output_size() == 1);
    Expects(element_type == output_element_type);
    Expects(num_inputs_ == GetInputIds().size());
    Expects(GetOutputIds().size() == 1);
    const auto& outputShape = concatOp.get_output_shape(0);
    const int64_t axis = concatOp.get_axis();
    Expects(axis >= 0 && axis < outputShape.size());
    auto num_chunks =
        std::accumulate(outputShape.begin(), outputShape.begin() + axis + 1, 1, std::multiplies<size_t>());
    Expects(num_chunks != 0);
    const std::size_t chunk_size =
        std::accumulate(outputShape.begin() + axis + 1, outputShape.end(), 1, std::multiplies<size_t>());
    Ensures(chunk_size != 0);
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
    concat_kernel_ = kernel::Concat{convertDataType<CUDAPlugin::kernel::Type_t>(element_type),
                                    num_inputs_,
                                    std::move(chunks),
                                    chunk_size,
                                    allChunkSize,
                                    numBlocks,
                                    threadsPerBlock};
}

WorkbufferRequest ConcatOp::GetWorkBufferRequest() const { return {{immutableWbSize()}, {mutableWbSize()}}; }

void ConcatOp::InitSharedImmutableWorkbuffers(const Buffers& buffers) {
    Expects(buffers.size() == 1);
    CUDA::DefaultStream::stream().upload(buffers[0], concat_kernel_.value().immutableWbData(), immutableWbSize());
}

void ConcatOp::Execute(const InferenceRequestContext& context,
                       Inputs inputs,
                       Outputs outputs,
                       const Workbuffers& workbuffers) const {
    Expects(concat_kernel_);
    Expects(inputs.size() == num_inputs_);
    Expects(outputs.size() == 1);
    auto& threadContext = context.getThreadContext();
    auto& stream = threadContext.stream();

    Expects(workbuffers.immutable_buffers.size() == 1);
    Expects(workbuffers.mutable_buffers.size() == 1);

    stream.upload(workbuffers.mutable_buffers[0], inputs.data(), mutableWbSize());
    (*concat_kernel_)(stream.get(),
                      workbuffers.immutable_buffers[0].get(),
                      reinterpret_cast<const void* const*>(workbuffers.mutable_buffers[0].get()),
                      outputs[0].get());
}

OPERATION_REGISTER(ConcatOp, Concat);
}  // namespace CUDAPlugin
