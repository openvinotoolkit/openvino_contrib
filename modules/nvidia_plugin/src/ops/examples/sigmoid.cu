// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include <utility>
#include <vector>

#include "sigmoid.hpp"

namespace ov {
namespace nvidia_gpu {

static __global__ void sigmoid(const size_t inputSize, const float* x, float* y) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < inputSize) {
        y[i] = 1 / (1 + expf(-x[i]));
    }
}

SigmoidOp::SigmoidOp(const CreationContext& context,
                     const std::shared_ptr<ov::Node>& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    auto input_element_type = node->get_input_element_type(0);
    auto output_element_type = node->get_output_element_type(0);
    OPENVINO_ASSERT(input_element_type.is_real());
    OPENVINO_ASSERT(output_element_type.is_real());
    auto input_shape = node->get_input_shape(0);
    auto output_shape = node->get_output_shape(0);
    input_size_ = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());
    output_size_ = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<size_t>());
    const unsigned maxBlockSize = context.device().props().maxThreadsPerBlock;
    num_blocks_ = (input_size_ % maxBlockSize == 0) ? (input_size_ / maxBlockSize) : (input_size_ / maxBlockSize + 1);
    threads_per_block_ = (num_blocks_ == 1) ? input_size_ : maxBlockSize;
}

void SigmoidOp::Execute(const InferenceRequestContext& context,
                        Inputs inputs,
                        Outputs outputs,
                        const Workbuffers& workbuffers) {
    OPENVINO_ASSERT(inputs.size() == 1);
    OPENVINO_ASSERT(outputs.size() == 1);
    stream.run(num_blocks_,
               threads_per_block_,
               sigmoid,
               input_size_,
               static_cast<const float*>(inputs[0].get()),
               static_cast<float*>(outputs[0].get()));
}

OPERATION_REGISTER(SigmoidOp, Sigmoid);
}  // namespace nvidia_gpu
}  // namespace ov
