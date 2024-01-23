// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernels/strided_slice.hpp"

#include <fmt/format.h>

#include "openvino/core/axis_set.hpp"

#include <cuda_operation_registry.hpp>
#include <openvino/op/constant.hpp>

#include "converters.hpp"
#include "strided_slice.hpp"

namespace ov {
namespace nvidia_gpu {

namespace {
template <typename T>
ov::AxisSet convert_mask_to_axis_set(const std::vector<T>& mask) {
    ov::AxisSet axis_set;
    for (size_t i = 0; i < mask.size(); ++i) {
        if (mask[i] == 1) {
            axis_set.insert(i);
        }
    }
    return axis_set;
}

template <typename T>
void calc_matrix_sizes(const ov::Shape& shape, std::vector<T>& matrix) {
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

template <typename T>
StridedSliceOp<T>::StridedSliceOp(const CreationContext& context,
                                  const NodeOp& stridedSliceOp,
                                  IndexCollection&& inputIds,
                                  IndexCollection&& outputIds)
    : OperationBase(context, stridedSliceOp, std::move(inputIds), std::move(outputIds)),
      element_type_{stridedSliceOp.get_input_element_type(0)},
      element_type_integer_{stridedSliceOp.get_input_element_type(1)} {
    const auto begin_const = get_node_constant_values(stridedSliceOp.get_input_node_ptr(1));
    const auto end_const = get_node_constant_values(stridedSliceOp.get_input_node_ptr(2));
    const auto stride_const = get_node_constant_values(stridedSliceOp.get_input_node_ptr(3));
    make_slice_plan(stridedSliceOp.get_input_shape(0),
                    begin_const,
                    end_const,
                    stride_const,
                    convert_mask_to_axis_set(stridedSliceOp.get_begin_mask()),
                    convert_mask_to_axis_set(stridedSliceOp.get_end_mask()),
                    convert_mask_to_axis_set(stridedSliceOp.get_new_axis_mask()),
                    convert_mask_to_axis_set(stridedSliceOp.get_shrink_axis_mask()),
                    convert_mask_to_axis_set(stridedSliceOp.get_ellipsis_mask()));
    src_matrix_sizes_ = std::vector<T>(stridedSliceOp.get_input_shape(0).size(), 0);
    dst_matrix_sizes_ = std::vector<T>(slice_plan_.reshape_in_shape.size(), 0);
    calc_matrix_sizes(stridedSliceOp.get_input_shape(0), src_matrix_sizes_);
    calc_matrix_sizes(slice_plan_.reshape_in_shape, dst_matrix_sizes_);

    const auto& prop = context.device().props();
    max_threads_per_block_ = prop.maxThreadsPerBlock;
    blocks_number_ = 1 + dst_matrix_sizes_[0] / max_threads_per_block_;
    threads_per_block_ = (blocks_number_ == 1) ? dst_matrix_sizes_[0] : max_threads_per_block_;

    kernel_op_ = std::make_optional<kernel::StridedSliceKernelOp<T>>(src_matrix_sizes_,
                                                                     dst_matrix_sizes_,
                                                                     slice_plan_.reverse_axes,
                                                                     max_threads_per_block_,
                                                                     blocks_number_,
                                                                     threads_per_block_,
                                                                     convertDataType<kernel::Type_t>(element_type_),
                                                                     convertDataType<kernel::Type_t>(element_type_integer_));
}

template <typename T>
void StridedSliceOp<T>::Execute(const InferenceRequestContext& context,
                                Inputs inputs,
                                Outputs outputs,
                                const Workbuffers& workbuffers) const {
    OPENVINO_ASSERT(kernel_op_, "Node name: ", GetName());
    (*kernel_op_)(context.getThreadContext().stream().get(),
                static_cast<const T*>(workbuffers.immutable_buffers[0].get()),
                inputs[0].get(),
                static_cast<const T*>(workbuffers.immutable_buffers[2].get()),
                static_cast<const T*>(workbuffers.immutable_buffers[3].get()),
                static_cast<const T*>(workbuffers.immutable_buffers[4].get()),
                static_cast<const T*>(workbuffers.immutable_buffers[1].get()),
                outputs[0].get());
}

template <typename T>
CudaGraphCompatibility StridedSliceOp<T>::GetCudaGraphCompatibility() const {
    return CudaGraphCompatibility::FULL;
}

template <typename T>
WorkbufferRequest StridedSliceOp<T>::GetWorkBufferRequest() const {
    return {{size_bytes(src_matrix_sizes_),
             size_bytes(dst_matrix_sizes_),
             size_bytes(slice_plan_.begins),
             size_bytes(slice_plan_.ends),
             size_bytes(slice_plan_.strides)},
            {}};
}

template <typename T>
void StridedSliceOp<T>::InitSharedImmutableWorkbuffers(const Buffers& buffers) {
    upload_data_to_workbuffer(buffers[0], src_matrix_sizes_);
    upload_data_to_workbuffer(buffers[1], dst_matrix_sizes_);
    upload_data_to_workbuffer(buffers[2], slice_plan_.begins);
    upload_data_to_workbuffer(buffers[3], slice_plan_.ends);
    upload_data_to_workbuffer(buffers[4], slice_plan_.strides);
}

template <typename T>
void StridedSliceOp<T>::upload_data_to_workbuffer(CUDA::DevicePointer<void*> buffer, const std::vector<T>& data) {
    auto& stream = CUDA::DefaultStream::stream();
    stream.upload(buffer, data.data(), size_bytes(data));
}

template <typename T>
std::vector<T> StridedSliceOp<T>::get_node_constant_values(const ov::Node* node) const {
    auto constant = dynamic_cast<const ov::op::v0::Constant*>(node);
    OPENVINO_ASSERT(constant, "Node name: ", GetName());
    OPENVINO_ASSERT(ov::element::Type_t::i64 == node->get_element_type() ||
                    ov::element::Type_t::i32 == node->get_element_type(),
                    "Node name: ", GetName());
    return constant->get_vector<T>();
}

template <typename T>
void StridedSliceOp<T>::make_slice_plan(const Shape& input_shape,
                                        const std::vector<T>& begins,
                                        const std::vector<T>& ends,
                                        const std::vector<T>& strides,
                                        const AxisSet& lower_bounds_mask,
                                        const AxisSet& upper_bounds_mask,
                                        const AxisSet& new_axis_mask,
                                        const AxisSet& shrink_axis_mask,
                                        const AxisSet& ellipsis_mask) {
    OPENVINO_ASSERT(begins.size() == ends.size());
    OPENVINO_ASSERT(ends.size() == strides.size());
    size_t num_slice_indices = begins.size();

    size_t num_real_axes = 0;
    size_t num_shrink_axes = 0;
    size_t num_new_axes = 0;
    bool ellipsis_found = false;

    // Make a pass over the original slices to make sure there is at most one
    // ellipsis, and to count up the number of shrink axes, the number of
    // "newaxis"es, and the number of "real" axes (axes that are not newaxis
    // and are not the ellipsis).
    for (size_t i = 0; i < num_slice_indices; i++) {
        if (ellipsis_mask.count(i)) {
            OPENVINO_ASSERT(!ellipsis_found);
            ellipsis_found = true;
        } else if (new_axis_mask.count(i)) {
            num_new_axes++;
        } else {
            if (shrink_axis_mask.count(i)) {
                num_shrink_axes++;
            }
            num_real_axes++;
        }
    }

    OPENVINO_ASSERT(num_real_axes <= input_shape.size(), "num_real_axes=", num_real_axes, ", input_shape=", input_shape);

    // Figure out how many axes need to be inserted when the ellipsis (which
    // may be an implicit ellipsis at the end) is expanded.
    size_t ellipsis_size = input_shape.size() - num_real_axes;

    // Initialize our slice plan.
    slice_plan_.begins = std::vector<T>(num_real_axes + ellipsis_size);
    slice_plan_.ends = std::vector<T>(num_real_axes + ellipsis_size);
    slice_plan_.strides = std::vector<T>(num_real_axes + ellipsis_size);
    slice_plan_.reshape_in_shape = Shape(num_real_axes + ellipsis_size);
    slice_plan_.reshape_out_shape = Shape(num_new_axes + num_real_axes + ellipsis_size - num_shrink_axes);
    slice_plan_.reverse_axes = AxisSet{};

    // Begin a maddeningly delicate loop to desugar the original slice.
    //
    // * i_in is iterating over the axes of the input shape, which are also the axes of
    //     slice_plan_.reshape_in_shape.
    // * i_out is iterating over the axes of slice_plan_.reshape_out_shape
    size_t i_in = 0;
    size_t i_out = 0;

    // If no actual ellipsis exists, there is an "implicit" one at the end,
    // which we will handle after the looslice_plan_. So the logic is wrapped up here,
    // allowing it to be used both during and after the looslice_plan_.
    auto expand_ellipsis = [&]() {
        for (size_t i = 0; i < ellipsis_size; i++) {
            slice_plan_.begins[i_in] = 0;
            slice_plan_.ends[i_in] = T(input_shape[i_in]);
            slice_plan_.strides[i_in] = 1;
            slice_plan_.reshape_in_shape[i_in] = input_shape[i_in];
            slice_plan_.reshape_out_shape[i_out] = input_shape[i_in];

            i_in++;
            i_out++;
        }
    };

    for (size_t i = 0; i < num_slice_indices; i++) {
        // If this is a "newaxis", then reshape_out_shape will have a 1 here,
        // but reshape_in_shape will not.
        if (new_axis_mask.count(i)) {
            slice_plan_.reshape_out_shape[i_out] = 1;
            i_out++;
        }
        // If this is a "shrunken" axis, then reshape_in_shape will have a 1
        // here, but reshape_out_shape will not.
        else if (shrink_axis_mask.count(i)) {
            T begin = begins[i];

            // Note that clipping is not used for "shrunken" axes: an
            // out-of-bounds index is an error.
            OPENVINO_ASSERT(begin >= -(T(input_shape[i_in])) && begin < T(input_shape[i_in]));

            if (begin < 0) {
                begin += T(input_shape[i_in]);
            }
            slice_plan_.begins[i_in] = begin;
            slice_plan_.ends[i_in] = begin + 1;
            slice_plan_.strides[i_in] = 1;
            slice_plan_.reshape_in_shape[i_in] = 1;
            i_in++;
        }
        // If this is the ellipsis, expand it.
        else if (ellipsis_mask.count(i)) {
            expand_ellipsis();
        }
        // In other cases, we have a nice, ordinary (begin:end:stride) slice.
        // We need to adjust for begin/end being masked, and begin/end/stride
        // being negative or out of bounds.
        else {
            bool is_reverse = strides[i] < 0;

            // Adjust the beginning for from-the-right indexing, and clislice_plan_.
            T real_begin = begins[i];
            if (lower_bounds_mask.count(i)) {
                real_begin = (is_reverse ? int64_t(input_shape[i_in] - 1) : 0);
            } else if (real_begin < 0) {
                real_begin += int64_t(input_shape[i_in]);
            }
            T max_real_begin = T(input_shape[i_in]) - (is_reverse ? 1 : 0);
            real_begin = std::max(T(0), std::min(max_real_begin, real_begin));

            // Adjust the ending for from-the-right indexing, and clislice_plan_.
            T real_end = ends[i];
            if (upper_bounds_mask.count(i)) {
                real_end = (is_reverse ? -1 : int64_t(input_shape[i_in]));
            } else if (real_end < 0) {
                real_end += int64_t(input_shape[i_in]);
            }
            T min_real_end = (is_reverse ? -1 : 0);
            real_end = std::max(min_real_end, std::min(T(input_shape[i_in]), real_end));

            // Ensure stride is not zero, and adjust it for backwards slicing.
            OPENVINO_ASSERT(strides[i] != 0);
            T real_stride = std::abs(strides[i]);

            // Adjust for reversal if needed. This isn't quite as simple as swapping begin and
            // end, due to striding; we have to adjust the end point to be the _actual_ leftmost
            // element, in cases where the stride does not evenly divide the span between begin
            // and end.
            if (is_reverse) {
                real_end += std::max(T(0), real_begin - real_end - 1) % real_stride;
                std::swap(real_begin, real_end);
                real_begin++;
                real_end++;
                slice_plan_.reverse_axes.insert(i_out);
            }

            // nGraph's slice op does not like it when end < begin, so we truncate for that case
            // here.
            if (real_end < real_begin) {
                real_end = real_begin;
            }

            // Compute output dimension.
            size_t dim = (real_end <= real_begin ? 0 : size_t(real_end - real_begin - 1) / size_t(real_stride) + 1);
            slice_plan_.reshape_in_shape[i_in] = dim;
            slice_plan_.reshape_out_shape[i_out] = dim;

            auto slice_size = real_end - real_begin;
            if (slice_size > 0 && real_stride > slice_size)
                real_stride = slice_size;
            if (real_stride == slice_size) {
                real_end = real_begin + 1;
                real_stride = 1;
            }

            // Set up the begin/end/stride.
            slice_plan_.begins[i_in] = real_begin;
            slice_plan_.ends[i_in] = real_end;
            slice_plan_.strides[i_in] = real_stride;

            i_in++;
            i_out++;
        }
    }

    // If there was no ellipsis explicitly given, there is an implicit one at
    // the end (it might encompass zero axes, but that's fine).
    if (!ellipsis_found) {
        expand_ellipsis();
    }
    return;
}

static OperationBase::Ptr strided_slice_factory(const CreationContext& context,
                                                const std::shared_ptr<ov::Node>& in_node,
                                                OperationBase::IndexCollection&& inputIds,
                                                OperationBase::IndexCollection&& outputIds) {
    const OperationBase::IndexCollection inputs{inputIds};
    const OperationBase::IndexCollection outputs{outputIds};

    auto node = std::dynamic_pointer_cast<ov::op::v1::StridedSlice>(in_node);
    OPENVINO_ASSERT(node);
    ov::element::Type element_type_integer = node->get_input_element_type(1);
    for (size_t i = 1; i < node->inputs().size(); i++) {
        OPENVINO_ASSERT(node->input(i).get_element_type() == element_type_integer);
    }
    if (ov::element::i32 == element_type_integer) {
        return std::make_shared<StridedSliceOp<int32_t>>(
            context, *node, OperationBase::IndexCollection{inputs}, OperationBase::IndexCollection{outputs});
    } else if (ov::element::i64 == element_type_integer) {
        return std::make_shared<StridedSliceOp<int64_t>>(
            context, *node, OperationBase::IndexCollection{inputs}, OperationBase::IndexCollection{outputs});
    } else {
        throw_ov_exception(fmt::format("Input precision {} is not supported by StridedSliceOp!", element_type_integer.get_type_name()));
    }
}

OPERATION_REGISTER_FACTORY(strided_slice_factory, StridedSlice)

}  // namespace nvidia_gpu
}  // namespace ov
