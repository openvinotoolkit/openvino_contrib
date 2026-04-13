// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/detection_output.hpp>

#include "cuda_operation_base.hpp"
#include "kernels/detection_output.hpp"

namespace ov {
namespace nvidia_gpu {

class DetectionOutputOp : public OperationBase {
public:
    using NodeOp = ov::op::v0::DetectionOutput;
    DetectionOutputOp(const CreationContext& context,
                      const NodeOp& node,
                      IndexCollection&& inputIds,
                      IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibilityImpl() const override;

    void InitSharedImmutableWorkbuffers(const Buffers& buffers) override;
    WorkbufferRequest GetWorkBufferRequest() const override;

private:
    const ov::element::Type element_type_;
    std::optional<kernel::DetectionOutput> kernel_;
};

}  // namespace nvidia_gpu
}  // namespace ov
