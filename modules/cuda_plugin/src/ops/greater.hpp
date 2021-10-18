// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cudnn_ops_infer.h>

#include <cuda/device_pointers.hpp>
#include <cuda_operation_base.hpp>
#include <gpu/gpu_context_api_cuda.hpp>
#include <kernels/greater.hpp>
#include <ngraph/op/greater.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/type/element_type.hpp>

namespace CUDAPlugin {

class GreaterOp : public OperationBase {
public:
    using NodeOp = ngraph::op::v1::Greater;
    GreaterOp(const CreationContext& context,
              const NodeOp& node,
              IndexCollection&& inputIds,
              IndexCollection&& outputIds);
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    void InitSharedImmutableWorkbuffers(const Buffers& buffers) override;

private:
    void calculateOffsets();
    WorkbufferRequest GetWorkBufferRequest() const override;

private:
    std::vector<size_t> output_shape_;
    std::vector<size_t> output_offset_;
    size_t max_size_;
    std::vector<size_t> output_sizes_;

    using Input = std::vector<size_t>;
    std::vector<Input> input_shapes_;
    std::vector<Input> input_offsets_;

    std::optional<kernel::Greater> kernel_;
};

}  // namespace CUDAPlugin
