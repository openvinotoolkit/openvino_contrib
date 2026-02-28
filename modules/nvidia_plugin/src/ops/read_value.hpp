// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include <string>

namespace ov {
namespace nvidia_gpu {

class ReadValueOp : public OperationBase {
public:
    ReadValueOp(const CreationContext& context,
                const ov::Node& node,
                IndexCollection&& inputIds,
                IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibility() const override {
        return CudaGraphCompatibility::NONE;
    }

    const std::string& GetVariableId() const { return variable_id_; }

private:
    std::string variable_id_;
    ov::element::Type element_type_;
    std::size_t output_byte_size_;
    bool has_init_value_;
};

}  // namespace nvidia_gpu
}  // namespace ov
