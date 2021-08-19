// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <kernels/saxpy.cuh>
#include <cuda_operation_registry.hpp>

#include "details/cuda_ngraph_import.hpp"
#include "saxpy_op.hpp"

namespace CUDAPlugin {

SaxpyOp::SaxpyOp(const CUDA::CreationContext& context,
                     const std::shared_ptr<ngraph::Node>& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    const unsigned max_block_size = context.device().props().maxThreadsPerBlock;
    const unsigned grid_size = kSize / max_block_size;
    const unsigned block_size = grid_size > 1 ? max_block_size : kSize % max_block_size;
    grid_dim_ = dim3{grid_size ? grid_size : 1};
    block_dim_ = dim3{block_size ? block_size : max_block_size};
}

void SaxpyOp::Execute(const InferenceRequestContext& context,
                      Inputs inputTensors,
                      Outputs outputTensors,
                      const Workbuffers&) {
    saxpy<<<grid_dim_, block_dim_>>>(
        kSize,
        inputTensors[0].cast<const float*>().get(),
        inputTensors[1].cast<const float*>().get(),
        outputTensors[0].cast<float*>().get());
}

OPERATION_REGISTER(SaxpyOp, Saxpy);

} // namespace CUDAPlugin
