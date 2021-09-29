// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include <ngraph/op/strided_slice.hpp>

#include "ngraph/slice_plan.hpp"

namespace CUDAPlugin {

class StridedSliceOp : public OperationBase {
public:
    using NodeOp = ngraph::op::v1::StridedSlice;
    StridedSliceOp(const CreationContext& context,
                   const NodeOp& stridedSliceOp,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds);
    void Execute(const InferenceRequestContext& context,
                 Inputs inputs,
                 Outputs outputs,
                 const Workbuffers& workbuffers) const override;
    WorkbufferRequest GetWorkBufferRequest() const override;
    void InitSharedImmutableWorkbuffers(const Buffers& buffers) override;

private:
    template <typename T>
    void callKernels(const InferenceRequestContext& context,
                     Inputs inputs,
                     Outputs outputs,
                     const Workbuffers& workbuffers) const;
    template <typename T>
    void callStridedSliceKernel(const InferenceRequestContext& context,
                                const Inputs inputs,
                                Outputs outputs,
                                const Workbuffers& workbuffers) const;
    template <typename T>
    void callReverseAxesKernel(const InferenceRequestContext& context, Outputs outputs) const;
    template <typename T>
    void callReverseAxesKernel(const InferenceRequestContext& context,
                               const std::vector<size_t>& matrixShapes,
                               const std::vector<int64_t>& matrixSizes,
                               const ngraph::AxisSet& reverseAxes,
                               CUDA::DevicePointer<void*> buffer) const;
    void uploadDataToWorkbuffer(CUDA::DevicePointer<void*> buffer, const std::vector<int64_t>& data);

    std::vector<int64_t> getNodeConstantValues(const ngraph::Node* node) const;

private:
    std::vector<int64_t> src_matrix_sizes;
    std::vector<int64_t> dst_matrix_sizes;

    ngraph::SlicePlan slice_plan;

    unsigned max_threads_per_block_{0};
    unsigned blocks_number_{0};
    unsigned threads_per_block_{0};
    ngraph::element::Type_t element_type_;
};

}  // namespace CUDAPlugin
