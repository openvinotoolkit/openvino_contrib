// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

#include "converters.hpp"
#include "kernels/variance_normalization_factor.hpp"
#include "openvino/op/mvn.hpp"

namespace ov {
namespace nvidia_gpu {

class MvnOp : public OperationCuDnn {
public:
    MvnOp(const CreationContext& context,
          const ov::Node& node,
          IndexCollection&& inputIds,
          IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibility() const override;
    WorkbufferRequest GetWorkBufferRequest() const override;

private:
    enum MvnVersion {
        MvnV1,
        MvnV6,
    };

    struct ConstTensor {
        const CUDA::DnnTensorDescriptor& descriptor;
        CUDA::DevicePointer<const void*> data;
    };
    struct Tensor {
        const CUDA::DnnTensorDescriptor& descriptor;
        CUDA::DevicePointer<void*> data;
    };
    struct Context {
        const InferenceRequestContext& context;
        const Workbuffers& workbuffers;
        const MvnOp& op;

        void reduceMean(ConstTensor input, Tensor output);
        void subtract(ConstTensor lhs, ConstTensor rhs, Tensor output);
        void multiply(ConstTensor lhs, ConstTensor rhs, Tensor output);
        void computeVarianceNormalizationFactor(Tensor in_out);
    };

    static MvnVersion validateAndGetVersion(const ov::Node& node);
    size_t reduceWorkSpaceSizeCompute(const CreationContext& context);
    ov::Shape makeReducedShape(const ov::Node& node);
    CUDA::DnnTensorDescriptor makeReducedTensorDescriptor(const ov::Node& node);
    CUDA::DeviceBuffer<std::uint8_t> getReduceWorkspaceBuffer(const Workbuffers& workbuffers) const {
        return workbuffers.createMutableSpanFrom<0>(reduce_workspace_size_);
    }
    CUDA::DevicePointer<void*> getReducedTensorBuffer(const Workbuffers& workbuffers) const {
        return workbuffers.mutable_buffers[1];
    }
    CUDA::DevicePointer<void*> getTmpTensorBuffer(const Workbuffers& workbuffers) const {
        return workbuffers.mutable_buffers[2];
    }

private:
    const ov::op::v0::MVN* mvn_op_v1_;
    const ov::op::v6::MVN* mvn_op_v6_;
    MvnVersion version_;
    bool normalize_variance_;
    double epsilon_;
    ov::op::MVNEpsMode eps_mode_;
    cudnnDataType_t comp_type_;
    cudnnDataType_t op_desc_type_;
    CUDA::DnnReduceAvgDescriptor reduce_mean_desc_;
    CUDA::DnnOpTensorDescriptor sub_desc_;
    CUDA::DnnOpTensorDescriptor mul_desc_;
    CUDA::DnnTensorDescriptor tensor_desc_;
    ov::Shape shape_;
    ov::Shape reduced_shape_;
    CUDA::DnnTensorDescriptor reduced_tensor_desc_;
    size_t reduce_workspace_size_;
    std::optional<kernel::VarianceNormalizationFactor> variance_normalization_factor_kernel_;
    const void* dOne{&CUDA::NumericConst<CUDA::constants::one>(comp_type_)};
    const void* dMinusOne{&CUDA::NumericConst<CUDA::constants::minusOne>(comp_type_)};
    const void* dZero{&CUDA::NumericConst<CUDA::constants::zero>(comp_type_)};
};

inline WorkbufferRequest MvnOp::GetWorkBufferRequest() const {
    if (!reduced_shape_.empty()) {
        if (normalize_variance_) {
            return {{},
                    {reduce_workspace_size_,
                     elementSize(comp_type_) * ov::shape_size(reduced_shape_),
                     elementSize(comp_type_) * ov::shape_size(shape_)}};
        } else {
            return {{}, {reduce_workspace_size_, elementSize(comp_type_) * ov::shape_size(reduced_shape_)}};
        }
    }
    return {};
}

}  // namespace nvidia_gpu
}  // namespace ov
