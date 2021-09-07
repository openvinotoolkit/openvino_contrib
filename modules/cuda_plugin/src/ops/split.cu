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
#include <ngraph/op/constant.hpp>
#include <ngraph/op/split.hpp>
#include <utility>
#include <vector>

#include "cuda/runtime.hpp"
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

SplitOp::SplitOp(const CUDA::CreationContext& context,
                 const ngraph::Node& node,
                 IndexCollection&& inputIds,
                 IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
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
        case ngraph::element::Type_t::undefined:
        case ngraph::element::Type_t::dynamic:
        case ngraph::element::Type_t::u1:
            throwIEException(fmt::format(
                "Input element type = {} is not supported by Split operation "
                "!!",
                static_cast<ngraph::element::Type_t>(input_element_type)));
    }
    element_type_ = input_element_type;

    auto& data_shape = splitOp->get_input_shape(0);
    const int64_t axis = *axisNode->get_data_ptr<int64_t>();
    Expects(axis >= 0 && axis < data_shape.size());
    Expects(data_shape[axis] % num_splits_ == 0);
    split_step_size_ = (data_shape[axis] / num_splits_) * std::accumulate(data_shape.begin()+axis+1, data_shape.end(), 1, std::multiplies<size_t>());
    Ensures(split_step_size_ != 0);
    num_split_chunks_ = std::accumulate(data_shape.begin(), data_shape.end(), 1, std::multiplies<size_t>()) / split_step_size_;
    Ensures(num_split_chunks_ != 0);
    const unsigned max_block_size = context.device().props().maxThreadsPerBlock;
    num_blocks_ = (num_split_chunks_ % max_block_size == 0) ?
                (num_split_chunks_ / max_block_size) :
                (num_split_chunks_ / max_block_size + 1);
    threads_per_block_ = (num_blocks_ == 1) ? num_split_chunks_ : max_block_size;
}

WorkbufferRequest SplitOp::GetWorkBufferRequest() const {
  return { {}, { mutableWbSize() } };
}

void SplitOp::Execute(const InferenceRequestContext& context, Inputs inputs, Outputs outputs, const Workbuffers& buffers) {
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
        default:
            throwIEException(fmt::format(
                "Input element type = {} is not supported by Split operation "
                "!!",
                static_cast<ngraph::element::Type_t>(element_type_)));
    }
}

template <typename T>
void SplitOp::Execute(const InferenceRequestContext& context, Inputs inputs, Outputs outputs, const Workbuffers& buffers) {
    Expects(inputs.size() == 2);
    Expects(outputs.size() == num_splits_);
    Expects(buffers.mutable_buffers.size() == 1);
    auto& threadContext = context.getThreadContext();
    auto& stream = threadContext.stream();
    auto outputPtrs = buffers.mutable_buffers[0];
    stream.upload(outputPtrs, reinterpret_cast<T**>(outputs.data()),
                  sizeof(T*) * num_splits_);
    auto in = inputs[0];
    split<T><<<num_blocks_, threads_per_block_, 0, stream.get()>>>(
        num_split_chunks_,
        split_step_size_,
        num_splits_,
        static_cast<const T *>(in.get()),
        reinterpret_cast<T **>(outputPtrs.get()));
}

OPERATION_REGISTER(SplitOp, Split);
} // namespace CUDAPlugin
