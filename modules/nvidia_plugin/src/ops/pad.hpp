// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cuda_operation_base.hpp>
#include <kernels/pad.cuh>
#include <openvino/op/pad.hpp>

namespace ov {
namespace nvidia_gpu {

class PadOp : public OperationBase {
public:
    using NodeOp = ov::op::v1::Pad;
    explicit PadOp(const CreationContext& context,
                   const ov::op::v1::Pad& node,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibility() const override;
    WorkbufferRequest GetWorkBufferRequest() const override;
    void InitSharedImmutableWorkbuffers(const Buffers&) override;

private:
    enum WorkbufferIndex {
        kSrcShape,
        kDstShape,
    };

    enum InputIndex {
        kSrc,
        kPadsBegin,
        kPadsEnd,
        kPadValue,
    };

    enum OutputIndex {
        kDst,
    };

    kernel::ConstModePad kernel_;
    ov::Shape src_shape_;
    ov::Shape dst_shape_;
};

}  // namespace nvidia_gpu
}  // namespace ov
