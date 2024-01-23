// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

namespace ov {
namespace nvidia_gpu {

class CuDnnTensorOpBase : public OperationCuDnn {
public:
    static constexpr std::size_t max_supported_shape_size = 5;

    CuDnnTensorOpBase(const CreationContext& context,
                      const std::shared_ptr<ov::Node>& node,
                      IndexCollection&& inputIds,
                      IndexCollection&& outputIds,
                      const cudnnOpTensorOp_t& opType,
                      const cudnnNanPropagation_t& nanPropogationType = cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN);
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibility() const override;

private:
    struct IoParams {
        const cudnnDataType_t type_;
        const ov::Shape shape_;
        std::array<int, 5> array_;
        CUDA::DnnTensorDescriptor desc_;
        enum class Type { INPUT, OUTPUT };

        IoParams(const ov::Node& node, const Type& io_type, int index);
    };

    static CUDA::DnnOpTensorDescriptor makeDnnOpTensorDescriptor(cudnnOpTensorOp_t opType,
                                                                 cudnnDataType_t dataType,
                                                                 cudnnNanPropagation_t nanPropogationType) {
        return CUDA::DnnOpTensorDescriptor{}.set(opType, dataType, nanPropogationType);
    }

    IoParams in0;
    IoParams in1;
    IoParams out;
    CUDA::DnnOpTensorDescriptor op_desc_;
    cudnnOpTensorOp_t op_type_;
    int bias_index_ = 0;
    int dest_index_ = 1;
};
}  // namespace nvidia_gpu
}  // namespace ov
