// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cudnn_ops_infer.h>
#include <ngraph/type/element_type.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/op/softmax.hpp>
#include <cuda_operation_base.hpp>
#include <gpu/device_pointers.hpp>
#include <gpu/gpu_context_api_cuda.hpp>

namespace CUDAPlugin {

class SplitOp : public OperationBase {
 public:
    SplitOp(const ngraph::Node& node,
            IndexCollection&& inputIds,
            IndexCollection&& outputIds);
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors) override;

 private:
    template <typename T>
    void callKernel(unsigned numBlocks, unsigned threadsPerBlock,
                    const CUDA::Stream& stream,
                    InferenceEngine::gpu::DevicePointer<const void*> in0,
                    const CUDA::Allocation& outputPtrs);

    ngraph::element::Type_t element_type_;
    size_t num_splits_ = 0;
    size_t num_split_chunks_ = 0;
    size_t split_step_size_ = 0;
};

} // namespace CUDAPlugin
