// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "split.hpp"

#include <fmt/format.h>

#include <cuda_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/split.hpp>
#include <utility>
#include <vector>

#include "converters.hpp"
#include "cuda/runtime.hpp"

namespace ov {
namespace nvidia_gpu {

SplitOp::SplitOp(const CreationContext& context,
                 const ov::Node& node,
                 IndexCollection&& inputIds,
                 IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    auto splitOp = dynamic_cast<const ov::op::v1::Split*>(&node);
    OPENVINO_ASSERT(splitOp, "Node name: ", GetName());
    auto input_element_type = splitOp->get_input_element_type(0);
    auto axisNode = dynamic_cast<ov::op::v0::Constant*>(splitOp->get_input_node_ptr(1));
    OPENVINO_ASSERT(axisNode, "Node name: ", GetName());
    auto output_element_type = splitOp->get_output_element_type(0);
    OPENVINO_ASSERT(splitOp->get_input_size() == 2, "Node name: ", GetName());
    num_splits_ = splitOp->get_num_splits();
    OPENVINO_ASSERT(num_splits_ != 0, "Node name: ", GetName());
    OPENVINO_ASSERT(splitOp->get_output_size() == num_splits_, "Node name: ", GetName());
    OPENVINO_ASSERT(input_element_type == output_element_type, "Node name: ", GetName());
    switch (input_element_type) {
        case ov::element::Type_t::dynamic:
        case ov::element::Type_t::u1:
            throw_ov_exception(
                fmt::format("Input element type = {} is not supported by Split operation "
                            "!!",
                            static_cast<ov::element::Type_t>(input_element_type)));
    }
    const auto element_type = input_element_type;

    auto& data_shape = splitOp->get_input_shape(0);
    const int64_t axis = *axisNode->get_data_ptr<int64_t>();
    OPENVINO_ASSERT(axis >= 0 && axis < data_shape.size(), "Node name: ", GetName());
    OPENVINO_ASSERT(data_shape[axis] % num_splits_ == 0, "Node name: ", GetName());
    const size_t split_step_size =
        (data_shape[axis] / num_splits_) *
        std::accumulate(data_shape.begin() + axis + 1, data_shape.end(), 1, std::multiplies<size_t>());
    OPENVINO_ASSERT(split_step_size != 0, "Node name: ", GetName());
    const size_t num_split_chunks =
        std::accumulate(data_shape.begin(), data_shape.end(), 1, std::multiplies<size_t>()) / split_step_size;
    OPENVINO_ASSERT(num_split_chunks != 0, "Node name: ", GetName());
    const size_t num_split_elements = split_step_size * num_split_chunks;
    const unsigned max_block_size = context.device().props().maxThreadsPerBlock;
    const unsigned num_blocks = (num_split_elements % max_block_size == 0) ? (num_split_elements / max_block_size)
                                                                           : (num_split_elements / max_block_size + 1);
    const unsigned threads_per_block = (num_blocks == 1) ? num_split_elements : max_block_size;

    split_kernel_ = kernel::Split{convertDataType<ov::nvidia_gpu::kernel::Type_t>(element_type),
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
    OPENVINO_ASSERT(split_kernel_, "Node name: ", GetName());
    OPENVINO_ASSERT(inputs.size() == 2, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == num_splits_, "Node name: ", GetName());
    OPENVINO_ASSERT(buffers.mutable_buffers.size() == 1, "Node name: ", GetName());
    auto& threadContext = context.getThreadContext();
    auto& stream = threadContext.stream();
    auto outputPtrs = buffers.mutable_buffers[0];
    stream.upload(outputPtrs, outputs.data(), sizeof(void*) * num_splits_);
    auto in = inputs[0];
    (*split_kernel_)(stream.get(), reinterpret_cast<const void*>(in.get()), reinterpret_cast<void**>(outputPtrs.get()));
}

CudaGraphCompatibility SplitOp::GetCudaGraphCompatibilityImpl() const { return CudaGraphCompatibility::NONE; }

OPERATION_REGISTER(SplitOp, Split);
}  // namespace nvidia_gpu
}  // namespace ov
