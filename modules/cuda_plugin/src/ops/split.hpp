// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cudnn_ops_infer.h>

#include <cuda_operation_base.hpp>
#include <gpu/device_pointers.hpp>
#include <gpu/gpu_context_api_cuda.hpp>
#include <kernels/split.hpp>
#include <ngraph/op/softmax.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/type/element_type.hpp>

namespace CUDAPlugin {

class SplitOp : public OperationBase {
public:
    SplitOp(const CUDA::CreationContext& context,
            const ngraph::Node& node,
            IndexCollection&& inputIds,
            IndexCollection&& outputIds);
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;
    WorkbufferRequest GetWorkBufferRequest() const override;

private:
    [[nodiscard]] size_t mutableWbSize() const { return split_kernel_.value().mutableWbSize(); }
    template <typename T>
    void Execute(const InferenceRequestContext& context,
                 Inputs inputs,
                 Outputs outputs,
                 const Workbuffers& buffers) const;

    size_t num_splits_ = 0;
    std::optional<kernel::Split> split_kernel_;
};

}  // namespace CUDAPlugin
