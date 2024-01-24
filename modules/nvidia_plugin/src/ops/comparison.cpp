// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "comparison.hpp"

#include <cuda_operation_registry.hpp>
#include <openvino/core/except.hpp>

#include "converters.hpp"

namespace ov {
namespace nvidia_gpu {

static constexpr auto kNumOfDim = 5u;
static constexpr auto kOffsetBufferSize = kNumOfDim * sizeof(size_t);

namespace {
enum InputIdx { LEFT, RIGHT, SIZES };

void calcOutOffset(std::vector<size_t>& offset, const std::vector<size_t>& dims) {
    offset.resize(kNumOfDim);
    size_t k = 1;
    auto i = dims.size();
    while (i >= 1) {
        auto j = i - 1;
        offset[j] = k;
        k *= dims[j];
        --i;
    }
}

void calcInOffset(std::vector<size_t>& offset, const std::vector<size_t>& inDims, const std::vector<size_t>& outDims) {
    offset.resize(kNumOfDim);
    size_t k = 1;
    auto i = inDims.size();
    while (i >= 1) {
        auto j = i - 1;
        offset[j] = (inDims[j] == outDims[j]) ? k : 0;
        k *= inDims[j];
        --i;
    }
}

}  // namespace

Comparison::Comparison(const CreationContext& context,
                       const ov::Node& node,
                       IndexCollection&& inputIds,
                       IndexCollection&& outputIds,
                       kernel::Comparison::Op_t operation_type)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)), output_shape_{node.get_output_shape(0)} {
    const ov::element::Type element_type{node.get_input_element_type(0)};
    auto output_element_type = node.get_output_element_type(0);

    for (auto i = 0u; i < node.get_input_size(); ++i) {
        input_shapes_.push_back(node.get_input_shape(i));
    }
    calculateOffsets();

    assert(output_offset_[0] > 0);

    for (size_t idx = 0; idx < output_offset_.size(); ++idx) {
        output_sizes_.push_back(output_offset_[idx] > 0 ? output_offset_[idx] : output_offset_[idx - 1]);
    }

    OPENVINO_ASSERT(output_element_type == ov::element::Type_t::boolean, "Node name: ", GetName());
    OPENVINO_ASSERT(node.get_output_size() == 1, "Node name: ", GetName());
    OPENVINO_ASSERT(node.get_input_size() == 2, "Node name: ", GetName());
    OPENVINO_ASSERT(GetOutputIds().size() == 1, "Node name: ", GetName());
    OPENVINO_ASSERT(GetInputIds().size() == 2, "Node name: ", GetName());

    const size_t output_size = ov::shape_size(output_shape_);
    const auto max_block_size = static_cast<unsigned>(context.device().props().maxThreadsPerBlock);

    const auto num_blocks =
        (output_size % max_block_size == 0) ? (output_size / max_block_size) : (output_size / max_block_size + 1);
    const auto threads_per_block = (num_blocks == 1) ? output_size : max_block_size;

    kernel_ = kernel::Comparison{operation_type,
                                 convertDataType<ov::nvidia_gpu::kernel::Type_t>(element_type),
                                 output_size,
                                 num_blocks,
                                 threads_per_block};
}

CudaGraphCompatibility Comparison::GetCudaGraphCompatibility() const { return CudaGraphCompatibility::FULL; }

void Comparison::Execute(const InferenceRequestContext& context,
                         Inputs inputs,
                         Outputs outputs,
                         const Workbuffers& workbuffers) const {
    OPENVINO_ASSERT(kernel_, "Node name: ", GetName());
    OPENVINO_ASSERT(inputs.size() == 2, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());
    auto& threadContext = context.getThreadContext();
    auto& stream = threadContext.stream();

    (*kernel_)(stream.get(),
               inputs[LEFT].get(),
               inputs[RIGHT].get(),
               static_cast<const size_t*>(workbuffers.immutable_buffers[LEFT].get()),
               static_cast<const size_t*>(workbuffers.immutable_buffers[RIGHT].get()),
               static_cast<const size_t*>(workbuffers.immutable_buffers[SIZES].get()),
               outputs[0].get());
}

void Comparison::InitSharedImmutableWorkbuffers(const Buffers& buffers) {
    auto& stream = CUDA::DefaultStream::stream();
    stream.upload(buffers[LEFT], input_offsets_[LEFT].data(), kOffsetBufferSize);
    stream.upload(buffers[RIGHT], input_offsets_[RIGHT].data(), kOffsetBufferSize);
    stream.upload(buffers[SIZES], output_sizes_.data(), kOffsetBufferSize);
}

WorkbufferRequest Comparison::GetWorkBufferRequest() const {
    return {std::vector<WorkbufferRequest::size_in_bytes_t>(SIZES + 1, kOffsetBufferSize), {}};
}

void Comparison::calculateOffsets() {
    std::vector<size_t> result_dims(kNumOfDim, 1);
    std::copy(std::begin(output_shape_),
              std::end(output_shape_),
              std::begin(result_dims) + (kNumOfDim - output_shape_.size()));
    calcOutOffset(output_offset_, result_dims);
    OPENVINO_ASSERT(output_offset_.size() == kNumOfDim, "Node name: ", GetName());

    for (const auto& shape : input_shapes_) {
        std::vector<size_t> result_shape(kNumOfDim, 1);
        std::copy(std::begin(shape), std::end(shape), std::begin(result_shape) + (kNumOfDim - shape.size()));
        calcInOffset(input_offsets_.emplace_back(), result_shape, result_dims);
        OPENVINO_ASSERT(input_offsets_.back().size() == kNumOfDim, "Node name: ", GetName());
    }
}

}  // namespace nvidia_gpu
}  // namespace ov
