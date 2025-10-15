// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "select.hpp"

#include <fmt/format.h>

#include <cuda_operation_registry.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/select.hpp>

#include "converters.hpp"

namespace ov {
namespace nvidia_gpu {

static constexpr auto OUTPUT = 0u;
static constexpr auto kNumOfDim = 5u;
static constexpr auto kOffsetBufferSize = kNumOfDim * sizeof(kernel::SelectKernelOp::BrcstOffsetType);

namespace {

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

SelectOp::SelectOp(const CreationContext& context,
                   const ov::Node& node,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)),
      output_shape_{node.get_output_shape(OUTPUT)},
      max_size_{ov::shape_size(output_shape_)},
      operation_type_{node.get_output_element_type(OUTPUT)} {
    for (auto i = 0u; i < node.get_input_size(); ++i) {
        input_shapes_.push_back(node.get_input_shape(i));
    }
    calculateOffsets();

    assert(output_offset_[0] > 0);
    for (size_t idx = 0; idx < output_offset_.size(); ++idx) {
        output_sizes_.push_back(output_offset_[idx] > 0 ? output_offset_[idx] : output_offset_[idx - 1]);
    }
    const auto& prop = context.device().props();
    max_threads_per_block_ = prop.maxThreadsPerBlock;
    blocks_number_ = 1 + max_size_ / max_threads_per_block_;
    threads_per_block_ = (blocks_number_ == 1) ? max_size_ : max_threads_per_block_;

    kernel_op_ =
        std::make_optional<kernel::SelectKernelOp>(max_size_, blocks_number_, threads_per_block_, convertDataType<kernel::Type_t>(operation_type_));
}

void SelectOp::Execute(const InferenceRequestContext& context,
                       Inputs inputs,
                       Outputs outputs,
                       const Workbuffers& workbuffers) const {
    assert(kernel_op_);
    (*kernel_op_)(
        context.getThreadContext().stream().get(),
        static_cast<const bool*>(inputs[CONDITION].get()),
        inputs[THEN].get(),
        inputs[ELSE].get(),
        static_cast<const kernel::SelectKernelOp::BrcstOffsetType*>(workbuffers.immutable_buffers[CONDITION].get()),
        static_cast<const kernel::SelectKernelOp::BrcstOffsetType*>(workbuffers.immutable_buffers[THEN].get()),
        static_cast<const kernel::SelectKernelOp::BrcstOffsetType*>(workbuffers.immutable_buffers[ELSE].get()),
        static_cast<const kernel::SelectKernelOp::BrcstOffsetType*>(workbuffers.immutable_buffers[SIZES].get()),
        outputs[0].get());
}

CudaGraphCompatibility SelectOp::GetCudaGraphCompatibility() const { return CudaGraphCompatibility::FULL; }

WorkbufferRequest SelectOp::GetWorkBufferRequest() const {
    return {std::vector<WorkbufferRequest::size_in_bytes_t>(SIZES + 1, kOffsetBufferSize), {}};
}

void SelectOp::InitSharedImmutableWorkbuffers(const Buffers& buffers) {
    auto& stream = CUDA::DefaultStream::stream();
    stream.upload(buffers[CONDITION], input_offsets_[CONDITION].data(), kOffsetBufferSize);
    stream.upload(buffers[THEN], input_offsets_[THEN].data(), kOffsetBufferSize);
    stream.upload(buffers[ELSE], input_offsets_[ELSE].data(), kOffsetBufferSize);
    stream.upload(buffers[SIZES], output_sizes_.data(), kOffsetBufferSize);
}

void SelectOp::calculateOffsets() {
    std::vector<size_t> result_dims(kNumOfDim, 1);
    std::copy(std::begin(output_shape_),
              std::end(output_shape_),
              std::begin(result_dims) + (kNumOfDim - output_shape_.size()));
    calcOutOffset(output_offset_, result_dims);
    assert(output_offset_.size() == kNumOfDim);

    for (const auto& shape : input_shapes_) {
        std::vector<size_t> result_shape(kNumOfDim, 1);
        std::copy(std::begin(shape), std::end(shape), std::begin(result_shape) + (kNumOfDim - shape.size()));
        calcInOffset(input_offsets_.emplace_back(), result_shape, result_dims);
        assert(input_offsets_.back().size() == kNumOfDim);
    }
}

OPERATION_REGISTER(SelectOp, Select);
}  // namespace nvidia_gpu
}  // namespace ov
