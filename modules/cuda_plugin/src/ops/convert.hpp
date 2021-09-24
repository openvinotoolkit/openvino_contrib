// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda/runtime.hpp>
#include <cuda_operation_base.hpp>
#include <ngraph/type/element_type.hpp>

namespace CUDAPlugin {

class ConvertOp : public OperationBase {
public:
    ConvertOp(const CUDA::CreationContext& context,
              const std::shared_ptr<ngraph::Node>& node,
              IndexCollection&& inputIds,
              IndexCollection&& outputIds);
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;
    using Type_t = ngraph::element::Type_t;
    using convert_t = void (*)(const CUDA::Stream&,
                               size_t,
                               InferenceEngine::gpu::DevicePointer<void*>,
                               InferenceEngine::gpu::DevicePointer<const void*>,
                               unsigned,
                               unsigned);

private:
    static convert_t getConvertKernel(Type_t output_type, Type_t input_type);
    convert_t convert_kernel_;
    unsigned size_;
    unsigned num_blocks_;
    unsigned threads_per_block_;
};

}  // namespace CUDAPlugin
