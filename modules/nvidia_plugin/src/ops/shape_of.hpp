// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

namespace ov {
namespace nvidia_gpu {

class ShapeOfOp : public OperationBase {
public:
    ShapeOfOp(const CreationContext& context,
              const ov::Node& node,
              IndexCollection&& inputIds,
              IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibilityImpl() const override {
        return CudaGraphCompatibility::NONE;
    }

private:
    ov::element::Type output_type_;
    ov::Shape static_input_shape_;
    bool is_dynamic_;
    std::size_t input_rank_;
};

}  // namespace nvidia_gpu
}  // namespace ov
