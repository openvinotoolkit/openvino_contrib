// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <ngraph/op/detection_output.hpp>

#include "cuda_operation_base.hpp"
#include "kernels/detection_output.hpp"

namespace CUDAPlugin {

class DetectionOutputOp : public OperationBase {
public:
    using NodeOp = ngraph::op::DetectionOutput;
    DetectionOutputOp(const CreationContext& context,
                      const NodeOp& node,
                      IndexCollection&& inputIds,
                      IndexCollection&& outputIds);
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    WorkbufferRequest GetWorkBufferRequest() const override {
        return {{}, kernel_.value().getMutableWorkbufferSize()};  // Most operators do not need workbuffers
    }

private:
    const ngraph::element::Type element_type_;
    std::optional<kernel::DetectionOutput> kernel_;
};

}  // namespace CUDAPlugin
