// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda/device_pointers.hpp>
#include <cuda_operation_base.hpp>
#include <transformer/nodes/fully_connected.hpp>

#include "cuda/constant_factory.hpp"
#include "matmul.hpp"

namespace ov {
namespace nvidia_gpu {

class FullyConnectedOp : public OperationCuBlas {
public:
    using NodeOp = nodes::FullyConnected;
    FullyConnectedOp(const CreationContext& context,
                     const NodeOp& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds);
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibility() const override;

private:
    MatMulOp matmul_op_;
    size_t bias_size_ = 0;
    size_t batch_bias_count_ = 0;
};

}  // namespace nvidia_gpu
}  // namespace ov
