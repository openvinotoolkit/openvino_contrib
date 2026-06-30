// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include <optional>

#include "kernels/select.hpp"

namespace ov {
namespace nvidia_gpu {

class SelectOp : public OperationBase {
    enum InputIdx { CONDITION, THEN, ELSE, SIZES };

public:
    SelectOp(const CreationContext& context,
             const ov::Node& node,
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
    void calculateOffsets();

private:
    std::vector<size_t> output_shape_;
    std::vector<kernel::SelectKernelOp::BrcstOffsetType> output_offset_;
    size_t max_size_;
    std::vector<kernel::SelectKernelOp::BrcstOffsetType> output_sizes_;

    using Input = std::vector<kernel::SelectKernelOp::BrcstOffsetType>;
    std::vector<Input> input_shapes_;
    std::vector<Input> input_offsets_;

    unsigned max_threads_per_block_{0};
    unsigned blocks_number_{0};
    unsigned threads_per_block_{0};
    ov::element::Type_t operation_type_;

    std::optional<kernel::SelectKernelOp> kernel_op_;
};

}  // namespace nvidia_gpu
}  // namespace ov
