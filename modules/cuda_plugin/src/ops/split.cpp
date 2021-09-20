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

#include "converters.hpp"
#include "cuda/runtime.hpp"
#include "split.hpp"

namespace CUDAPlugin {

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
            throwIEException(
                fmt::format("Input element type = {} is not supported by Split operation "
                            "!!",
                            static_cast<ngraph::element::Type_t>(input_element_type)));
    }
    const auto element_type = input_element_type;

    auto& data_shape = splitOp->get_input_shape(0);
    const int64_t axis = *axisNode->get_data_ptr<int64_t>();
    Expects(axis >= 0 && axis < data_shape.size());
    Expects(data_shape[axis] % num_splits_ == 0);
    const size_t split_step_size =
        (data_shape[axis] / num_splits_) *
        std::accumulate(data_shape.begin() + axis + 1, data_shape.end(), 1, std::multiplies<size_t>());
    Ensures(split_step_size != 0);
    const size_t num_split_chunks =
        std::accumulate(data_shape.begin(), data_shape.end(), 1, std::multiplies<size_t>()) / split_step_size;
    Ensures(num_split_chunks != 0);
    const unsigned max_block_size = context.device().props().maxThreadsPerBlock;
    const unsigned num_blocks = (num_split_chunks % max_block_size == 0) ? (num_split_chunks / max_block_size)
                                                                         : (num_split_chunks / max_block_size + 1);
    const unsigned threads_per_block = (num_blocks == 1) ? num_split_chunks : max_block_size;

    split_kernel_ = kernel::Split{convertDataType<CUDAPlugin::kernel::Type_t>(element_type),
                                  num_splits_,
                                  num_split_chunks,
                                  split_step_size,
                                  num_blocks,
                                  threads_per_block};
}

WorkbufferRequest SplitOp::GetWorkBufferRequest() const { return {{}, {mutableWbSize()}}; }

void SplitOp::Execute(const InferenceRequestContext& context,
                      Inputs inputs,
                      Outputs outputs,
                      const Workbuffers& buffers) const {
    Expects(split_kernel_);
    Expects(inputs.size() == 2);
    Expects(outputs.size() == num_splits_);
    Expects(buffers.mutable_buffers.size() == 1);
    auto& threadContext = context.getThreadContext();
    auto& stream = threadContext.stream();
    auto outputPtrs = buffers.mutable_buffers[0];
    stream.upload(outputPtrs, outputs.data(), sizeof(void*) * num_splits_);
    auto in = inputs[0];
    (*split_kernel_)(stream.get(), reinterpret_cast<const void*>(in.get()), reinterpret_cast<void**>(outputPtrs.get()));
}

OPERATION_REGISTER(SplitOp, Split);
}  // namespace CUDAPlugin
