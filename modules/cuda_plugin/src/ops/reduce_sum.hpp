// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

namespace CUDAPlugin {

class ReduceSumOp : public OperationCuDnn {
public:
    ReduceSumOp(const CreationContext& context,
                const ov::Node& node,
                IndexCollection&& inputIds,
                IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    WorkbufferRequest GetWorkBufferRequest() const override;

private:
    cudnnDataType_t comp_type_;
    CUDA::DnnReduceAddDescriptor add_desc_{comp_type_};
    CUDA::DnnTensorDescriptor a_desc_;
    CUDA::DnnTensorDescriptor c_desc_;
    size_t workspace_size_;
};

inline WorkbufferRequest ReduceSumOp::GetWorkBufferRequest() const {
    return {{}, {workspace_size_}};  // TODO: find a way to allocate buffers from constructor
}

}  // namespace CUDAPlugin
