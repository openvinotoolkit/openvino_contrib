// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

namespace ov {
namespace nvidia_gpu {

class ReduceOp : public OperationCuDnn {
public:
    ReduceOp(const CreationContext& context,
             const ov::Node& node,
             IndexCollection&& inputIds,
             IndexCollection&& outputIds,
             const CUDA::DnnReduceTensorDescriptor& reduce_desc);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibility() const override;
    WorkbufferRequest GetWorkBufferRequest() const override;

    static cudnnDataType_t reduceCompType(const ov::Node& node);

private:
    cudnnDataType_t comp_type_;
    CUDA::DnnReduceTensorDescriptor reduce_desc_;
    CUDA::DnnTensorDescriptor a_desc_;
    CUDA::DnnTensorDescriptor c_desc_;
    size_t workspace_size_;
};

inline WorkbufferRequest ReduceOp::GetWorkBufferRequest() const {
    return {{}, {workspace_size_}};  // TODO: find a way to allocate buffers from constructor
}

}  // namespace nvidia_gpu
}  // namespace ov
