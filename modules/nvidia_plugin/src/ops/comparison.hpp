// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda_operation_base.hpp"
#include "kernels/comparison.hpp"

namespace ov {
namespace nvidia_gpu {

class Comparison : public OperationBase {
public:
    Comparison(const CreationContext& context,
               const ov::Node& node,
               IndexCollection&& inputIds,
               IndexCollection&& outputIds,
               kernel::Comparison::Op_t operation_type);

    CudaGraphCompatibility GetCudaGraphCompatibility() const override;

private:
    void calculateOffsets();
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override final;

    void InitSharedImmutableWorkbuffers(const Buffers& buffers) override final;
    WorkbufferRequest GetWorkBufferRequest() const override final;

private:
    std::vector<size_t> output_shape_;
    std::vector<size_t> output_offset_;
    std::vector<size_t> output_sizes_;

    using Input = std::vector<size_t>;
    std::vector<Input> input_shapes_;
    std::vector<Input> input_offsets_;

    std::optional<kernel::Comparison> kernel_;
};

}  // namespace nvidia_gpu
}  // namespace ov
