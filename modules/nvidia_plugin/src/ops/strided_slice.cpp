// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernels/strided_slice.hpp"

#include <fmt/format.h>

#include <cuda_operation_registry.hpp>
#include <openvino/op/constant.hpp>

#include "converters.hpp"
#include "ngraph/axis_set.hpp"
#include "strided_slice.hpp"

namespace ov {
namespace nvidia_gpu {

namespace {
ov::AxisSet convert_mask_to_axis_set(const std::vector<int64_t>& mask) {
    ov::AxisSet axis_set;
    for (size_t i = 0; i < mask.size(); ++i) {
        if (mask[i] == 1) {
            axis_set.insert(i);
        }
    }
    return axis_set;
}

void calcMatrixSizes(const ov::Shape& shape, std::vector<int64_t>& matrix) {
    size_t prev_shape_size = 1;

    for (size_t src_shape_idx = shape.size(); src_shape_idx > 0; --src_shape_idx) {
        prev_shape_size = shape[src_shape_idx - 1] * prev_shape_size;
        matrix[src_shape_idx - 1] = prev_shape_size;
    }
}

template <typename T>
auto size_bytes(const std::vector<T>& v) noexcept {
    return sizeof(T) * v.size();
}

}  // namespace

StridedSliceOp::StridedSliceOp(const CreationContext& context,
                               const NodeOp& stridedSliceOp,
                               IndexCollection&& inputIds,
                               IndexCollection&& outputIds)
    : OperationBase(context, stridedSliceOp, std::move(inputIds), std::move(outputIds)),
      element_type_{stridedSliceOp.get_input_element_type(0)} {
    for (size_t i = 1; i < stridedSliceOp.inputs().size(); i++) {
        if (stridedSliceOp.input(i).get_element_type() != ov::element::Type_t::i64) {
            throwIEException(fmt::format("Input precision {} is not supported by StridedSliceOp!",
                                         stridedSliceOp.input(i).get_element_type().get_type_name()));
        }
    }

    const auto begin_const = getNodeConstantValues(stridedSliceOp.get_input_node_ptr(1));
    const auto end_const = getNodeConstantValues(stridedSliceOp.get_input_node_ptr(2));
    const auto stride_const = getNodeConstantValues(stridedSliceOp.get_input_node_ptr(3));
    slice_plan_ = make_slice_plan(stridedSliceOp.get_input_shape(0),
                                  begin_const,
                                  end_const,
                                  stride_const,
                                  convert_mask_to_axis_set(stridedSliceOp.get_begin_mask()),
                                  convert_mask_to_axis_set(stridedSliceOp.get_end_mask()),
                                  convert_mask_to_axis_set(stridedSliceOp.get_new_axis_mask()),
                                  convert_mask_to_axis_set(stridedSliceOp.get_shrink_axis_mask()),
                                  convert_mask_to_axis_set(stridedSliceOp.get_ellipsis_mask()));
    src_matrix_sizes_ = std::vector<int64_t>(stridedSliceOp.get_input_shape(0).size(), 0);
    dst_matrix_sizes_ = std::vector<int64_t>(slice_plan_.reshape_in_shape.size(), 0);
    calcMatrixSizes(stridedSliceOp.get_input_shape(0), src_matrix_sizes_);
    calcMatrixSizes(slice_plan_.reshape_in_shape, dst_matrix_sizes_);

    const auto& prop = context.device().props();
    max_threads_per_block_ = prop.maxThreadsPerBlock;
    blocks_number_ = 1 + dst_matrix_sizes_[0] / max_threads_per_block_;
    threads_per_block_ = (blocks_number_ == 1) ? dst_matrix_sizes_[0] : max_threads_per_block_;

    kernel_op_ = std::make_optional<kernel::StridedSliceKernelOp>(src_matrix_sizes_,
                                                                  dst_matrix_sizes_,
                                                                  slice_plan_.reverse_axes,
                                                                  max_threads_per_block_,
                                                                  blocks_number_,
                                                                  threads_per_block_,
                                                                  convertDataType<kernel::Type_t>(element_type_));
}

void StridedSliceOp::Execute(const InferenceRequestContext& context,
                             Inputs inputs,
                             Outputs outputs,
                             const Workbuffers& workbuffers) const {
    Expects(kernel_op_);
    (*kernel_op_)(context.getThreadContext().stream().get(),
                  static_cast<const int64_t*>(workbuffers.immutable_buffers[0].get()),
                  inputs[0].get(),
                  static_cast<const int64_t*>(workbuffers.immutable_buffers[2].get()),
                  static_cast<const int64_t*>(workbuffers.immutable_buffers[3].get()),
                  static_cast<const int64_t*>(workbuffers.immutable_buffers[4].get()),
                  static_cast<const int64_t*>(workbuffers.immutable_buffers[1].get()),
                  outputs[0].get());
}

WorkbufferRequest StridedSliceOp::GetWorkBufferRequest() const {
    return {{size_bytes(src_matrix_sizes_),
             size_bytes(dst_matrix_sizes_),
             size_bytes(slice_plan_.begins),
             size_bytes(slice_plan_.ends),
             size_bytes(slice_plan_.strides)},
            {}};
}

void StridedSliceOp::InitSharedImmutableWorkbuffers(const Buffers& buffers) {
    uploadDataToWorkbuffer(buffers[0], src_matrix_sizes_);
    uploadDataToWorkbuffer(buffers[1], dst_matrix_sizes_);

    uploadDataToWorkbuffer(buffers[2], slice_plan_.begins);
    uploadDataToWorkbuffer(buffers[3], slice_plan_.ends);
    uploadDataToWorkbuffer(buffers[4], slice_plan_.strides);
}

void StridedSliceOp::uploadDataToWorkbuffer(CUDA::DevicePointer<void*> buffer, const std::vector<int64_t>& data) {
    auto& stream = CUDA::DefaultStream::stream();
    stream.upload(buffer, data.data(), size_bytes(data));
}

std::vector<int64_t> StridedSliceOp::getNodeConstantValues(const ov::Node* node) const {
    auto constant = dynamic_cast<const ov::op::v0::Constant*>(node);
    Expects(constant);
    Expects(ov::element::Type_t::i64 == node->get_element_type());
    const int64_t* begin = constant->get_data_ptr<int64_t>();
    return std::vector<int64_t>(begin, begin + shape_size(constant->get_shape()));
}

OPERATION_REGISTER(StridedSliceOp, StridedSlice);

}  // namespace nvidia_gpu
}  // namespace ov
