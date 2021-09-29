// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda/device_pointers.hpp>
#include <cuda_operation_base.hpp>
#include <transformer/nodes/fully_connected.hpp>

#include "constant_factory.hpp"
#include "matmul.hpp"

namespace CUDAPlugin {

class FullyConnectedOp : public OperationCuBlas {
public:
    using NodeOp = nodes::FullyConnected;
    FullyConnectedOp(const CUDA::CreationContext& context,
                     const NodeOp& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds);
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

private:
    MatMulOp matmul_op_;
    size_t bias_size_ = 0;
    int batch_bias_count_ = 0;
};

}  // namespace CUDAPlugin
