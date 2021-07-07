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
#include "split.hpp"

namespace CUDAPlugin {

template <typename T>
static __global__ void split(const size_t numSplitChunks,
                             const size_t splitStepSize,
                             const size_t numSplits,
                             const T *x,
                             T **y) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numSplitChunks) {
        const unsigned splitIdx = i % numSplits;
        const unsigned splitStepIdx = i / numSplits;
        auto src = &x[i*splitStepSize];
        auto dest = &y[splitIdx][splitStepIdx * splitStepSize];
        memcpy(dest, src, sizeof(T) * splitStepSize);
    }
}

SplitOp::SplitOp(const CUDA::Device& device,
                 const ngraph::Node& node,
                 IndexCollection&& inputIds,
                 IndexCollection&& outputIds)
    : OperationBase(device, node, std::move(inputIds), std::move(outputIds)) {
    auto splitOp = dynamic_cast<const ngraph::op::v1::Split*>(&node);
    Expects(splitOp);
    auto input_element_type = splitOp->get_input_element_type(0);
    auto axisNode = dynamic_cast<ngraph::op::v0::Constant*>(splitOp->get_input_node_ptr(1));
    Expects(axisNode);
    auto output_element_type = splitOp->get_output_element_type(0);
    Expects(splitOp->get_input_size() == 2);
    num_splits_ = splitOp->get_num_splits();
    Ensures(num_splits_ != 0);
    Expects(splitOp->get_output_size() == num_splits_);
    Expects(input_element_type == output_element_type);
    switch (input_element_type) {
        case ngraph::element::Type_t::boolean:
        case ngraph::element::Type_t::bf16:
        case ngraph::element::Type_t::f16:
        case ngraph::element::Type_t::f32:
        case ngraph::element::Type_t::f64:
        case ngraph::element::Type_t::i8:
        case ngraph::element::Type_t::i16:
        case ngraph::element::Type_t::i32:
        case ngraph::element::Type_t::i64:
        case ngraph::element::Type_t::u8:
        case ngraph::element::Type_t::u16:
        case ngraph::element::Type_t::u32:
        case ngraph::element::Type_t::u64:
            break;
        case ngraph::element::Type_t::undefined:
        case ngraph::element::Type_t::dynamic:
        case ngraph::element::Type_t::u1:
        default: THROW_IE_EXCEPTION << fmt::format("Input element type = {} is not supported by Split operation !!",
                                                   static_cast<ngraph::element::Type_t>(input_element_type));
    }
    element_type_ = input_element_type;

    auto& dataShape = splitOp->get_input_shape(0);
    const int64_t axis = *axisNode->get_data_ptr<int64_t>();
    Expects(axis >= 0 && axis < dataShape.size());
    Expects(dataShape[axis] % num_splits_ == 0);
    split_step_size_ = (dataShape[axis] / num_splits_) * std::accumulate(dataShape.begin()+axis+1, dataShape.end(), 1, std::multiplies<size_t>());
    Ensures(split_step_size_ != 0);
    num_split_chunks_ = std::accumulate(dataShape.begin(), dataShape.end(), 1, std::multiplies<size_t>()) / split_step_size_;
    Ensures(num_split_chunks_ != 0);
}

void SplitOp::Execute(const InferenceRequestContext& context, Inputs inputs, Outputs outputs, const Workbuffers&) {
    Expects(inputs.size() == 2);
    Expects(outputs.size() == num_splits_);
    auto& threadContext = context.getThreadContext();
    auto& stream = threadContext.stream();
    const unsigned maxBlockSize = threadContext.device().props().maxThreadsPerBlock;
    const unsigned numBlocks = (num_split_chunks_ % maxBlockSize == 0) ?
                               (num_split_chunks_ / maxBlockSize) :
                               (num_split_chunks_ / maxBlockSize + 1);
    const unsigned threadsPerBlock = (numBlocks == 1) ? num_split_chunks_ : maxBlockSize;
    const CUDA::Allocation outputPtrs = stream.malloc(sizeof(float *) * num_splits_);
    stream.upload(outputPtrs.get(), reinterpret_cast<float **>(outputs.data()), sizeof(float *) * num_splits_);
    auto in0 = inputs[0];
    switch (element_type_) {
        case ngraph::element::Type_t::boolean: return callKernel<bool>(numBlocks, threadsPerBlock, stream, in0, outputPtrs);
        case ngraph::element::Type_t::bf16: return callKernel<__nv_bfloat16>(numBlocks, threadsPerBlock, stream, in0, outputPtrs);
        case ngraph::element::Type_t::f16: return callKernel<__half>(numBlocks, threadsPerBlock, stream, in0, outputPtrs);
        case ngraph::element::Type_t::f32: return callKernel<float>(numBlocks, threadsPerBlock, stream, in0, outputPtrs);
        case ngraph::element::Type_t::f64: return callKernel<double>(numBlocks, threadsPerBlock, stream, in0, outputPtrs);
        case ngraph::element::Type_t::i8: return callKernel<int8_t>(numBlocks, threadsPerBlock, stream, in0, outputPtrs);
        case ngraph::element::Type_t::i16: return callKernel<int16_t>(numBlocks, threadsPerBlock, stream, in0, outputPtrs);
        case ngraph::element::Type_t::i32: return callKernel<int32_t>(numBlocks, threadsPerBlock, stream, in0, outputPtrs);
        case ngraph::element::Type_t::i64: return callKernel<int64_t>(numBlocks, threadsPerBlock, stream, in0, outputPtrs);
        case ngraph::element::Type_t::u8: return callKernel<uint8_t>(numBlocks, threadsPerBlock, stream, in0, outputPtrs);
        case ngraph::element::Type_t::u16: return callKernel<uint16_t>(numBlocks, threadsPerBlock, stream, in0, outputPtrs);
        case ngraph::element::Type_t::u32: return callKernel<uint32_t>(numBlocks, threadsPerBlock, stream, in0, outputPtrs);
        case ngraph::element::Type_t::u64: return callKernel<uint64_t>(numBlocks, threadsPerBlock, stream, in0, outputPtrs);
        case ngraph::element::Type_t::undefined:
        case ngraph::element::Type_t::dynamic:
        case ngraph::element::Type_t::u1:
        default: THROW_IE_EXCEPTION << fmt::format("Input element type = {} is not supported by Split operation !!",
                                                   static_cast<ngraph::element::Type_t>(element_type_));
    }
}

template <typename T>
void SplitOp::callKernel(const unsigned numBlocks, const unsigned threadsPerBlock,
                         const CUDA::Stream& stream,
                         InferenceEngine::gpu::DevicePointer<const void*> in0,
                         const CUDA::Allocation& outputPtrs) {
    split<T><<<numBlocks, threadsPerBlock, 0, stream.get()>>>(
        num_split_chunks_,
        split_step_size_,
        num_splits_,
        static_cast<const T *>(in0.get()),
        reinterpret_cast<T **>(outputPtrs.get()));
}

OPERATION_REGISTER(SplitOp, Split);
} // namespace CUDAPlugin
