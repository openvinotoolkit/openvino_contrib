// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cudnn_ops_infer_v8.h>

#include <cuda/dnn.hpp>
#include <cuda_operation_base.hpp>
#include <ngraph/op/clamp.hpp>

namespace CUDAPlugin {

class ClampCuDnnOp : public OperationCuDnn {
public:
    using NodeOp = ngraph::op::Clamp;

    static constexpr std::size_t max_shape_size = 5;

    ClampCuDnnOp(const CreationContext& context,
                 const NodeOp& node,
                 IndexCollection&& inputIds,
                 IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    void InitSharedImmutableWorkbuffers(const Buffers& buffers) override;
    WorkbufferRequest GetWorkBufferRequest() const override;

private:
    static constexpr int max_index = 0;
    static constexpr int min_index = 1;

    template <typename T>
    void initBuffers(const Buffers& buffers) const;

    const cudnnDataType_t data_type_;
    const cudnnDataType_t op_type_;
    const CUDA::DnnOpTensorDescriptor max_op_desc_;
    const CUDA::DnnOpTensorDescriptor min_op_desc_;
    const CUDA::DnnTensorDescriptor io_desc_;
    const CUDA::DnnTensorDescriptor max_min_desc_;
    const double max_;
    const double min_;
};
}  // namespace CUDAPlugin
