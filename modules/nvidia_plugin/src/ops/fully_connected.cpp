// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected.hpp"

#include <cuda/blas.hpp>
#include <cuda_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include <openvino/op/matmul.hpp>
#include <utility>

#include "converters.hpp"
#include "cuda/constant_factory.hpp"
#include "matmul.hpp"

namespace ov {
namespace nvidia_gpu {

FullyConnectedOp::FullyConnectedOp(const CreationContext& context,
                                   const NodeOp& node,
                                   IndexCollection&& inputIds,
                                   IndexCollection&& outputIds)
    : OperationCuBlas(context, node, std::move(inputIds), std::move(outputIds)),
      matmul_op_{
          context, node, IndexCollection{input_ids_.begin(), input_ids_.end() - 1}, IndexCollection(output_ids_)} {
    bias_size_ = node.get_input_tensor(2).size();
    auto biasShape = node.get_input_shape(2);
    auto matrixShape = node.get_output_shape(0);
    OPENVINO_ASSERT(biasShape.size() > 0, "Node name: ", GetName());
    MatMulOp::BroadcastToMatrix(biasShape);
    const auto biasShapeSize = ov::shape_size(biasShape);
    const auto matrixShapeSize = ov::shape_size(matrixShape);
    OPENVINO_ASSERT(matrixShapeSize >= biasShapeSize, "Node name: ", GetName());
    auto batchBiasCount = MatMulOp::GetMatrixNumBatches(biasShape);
    auto matMulBatchCount = matmul_op_.GetBatchCount();
    OPENVINO_ASSERT(matMulBatchCount >= batchBiasCount, "Node name: ", GetName());
    batch_bias_count_ = matrixShapeSize / biasShapeSize;
}

void FullyConnectedOp::Execute(const InferenceRequestContext& context,
                               Inputs inputs,
                               Outputs outputs,
                               const Workbuffers& workbuffers) const {
    OPENVINO_ASSERT(inputs.size() == 3, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());
    auto& stream = context.getThreadContext().stream();

    auto bias = inputs[2];
    auto matrixC = outputs[0];
    for (size_t i = 0; i < batch_bias_count_; ++i) {
        stream.transfer(matrixC + i * bias_size_, bias, bias_size_);
    }
    matmul_op_.Execute(context, inputs.first(inputs.size() - 1), outputs, workbuffers);
}

CudaGraphCompatibility FullyConnectedOp::GetCudaGraphCompatibilityImpl() const { return CudaGraphCompatibility::FULL; }

OPERATION_REGISTER(FullyConnectedOp, FullyConnected);
}  // namespace nvidia_gpu
}  // namespace ov
