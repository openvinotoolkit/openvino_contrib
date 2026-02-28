// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda_operation_base.hpp"
#include "kernels/strided_slice.hpp"
#include "openvino/op/strided_slice.hpp"

namespace ov {
namespace nvidia_gpu {

template <typename T>
class StridedSliceOp : public OperationBase {
public:
    using NodeOp = ov::op::v1::StridedSlice;
    struct SlicePlan {
        // Parameters for the Slice
        std::vector<T> begins;
        std::vector<T> ends;
        std::vector<T> strides;

        // Shapes coming into, and going out of, the Reshape.
        Shape reshape_in_shape;
        Shape reshape_out_shape;

        // Parameters for the Reverse
        AxisSet reverse_axes;
    };

    StridedSliceOp(const CreationContext& context,
                   const NodeOp& stridedSliceOp,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputs,
                 Outputs outputs,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibilityImpl() const override;
    WorkbufferRequest GetWorkBufferRequest() const override;
    void InitSharedImmutableWorkbuffers(const Buffers& buffers) override;

private:
    void upload_data_to_workbuffer(CUDA::DevicePointer<void*> buffer, const std::vector<T>& data);
    std::vector<T> get_node_constant_values(const ov::Node* node) const;
    void make_slice_plan(const Shape& input_shape,
                         const std::vector<T>& begins,
                         const std::vector<T>& ends,
                         const std::vector<T>& strides,
                         const AxisSet& lower_bounds_mask,
                         const AxisSet& upper_bounds_mask,
                         const AxisSet& new_axis_mask,
                         const AxisSet& shrink_axis_mask,
                         const AxisSet& ellipsis_mask);

    std::vector<T> src_matrix_sizes_;
    std::vector<T> dst_matrix_sizes_;

    SlicePlan slice_plan_;

    unsigned max_threads_per_block_{0};
    unsigned blocks_number_{0};
    unsigned threads_per_block_{0};
    ov::element::Type_t element_type_;
    ov::element::Type_t element_type_integer_;

    std::optional<kernel::StridedSliceKernelOp<T>> kernel_op_;
};
}  // namespace nvidia_gpu
}  // namespace ov
