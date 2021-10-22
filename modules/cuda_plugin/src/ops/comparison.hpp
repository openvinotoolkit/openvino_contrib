// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>

#include "cuda_operation_base.hpp"
#include "kernels/comparison.hpp"

namespace CUDAPlugin {

class Comparison : public OperationBase {
public:
    Comparison(const CreationContext& context,
               const ngraph::Node& node,
               IndexCollection&& inputIds,
               IndexCollection&& outputIds,
               kernel::Comparison::Op_t operation_type);

private:
    void calculateOffsets();
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const final;

    void InitSharedImmutableWorkbuffers(const Buffers& buffers) final;
    WorkbufferRequest GetWorkBufferRequest() const final;

private:
    std::vector<size_t> output_shape_;
    std::vector<size_t> output_offset_;
    size_t max_size_;
    std::vector<size_t> output_sizes_;

    using Input = std::vector<size_t>;
    std::vector<Input> input_shapes_;
    std::vector<Input> input_offsets_;

    std::optional<kernel::Comparison> kernel_;
};

}  // namespace CUDAPlugin
