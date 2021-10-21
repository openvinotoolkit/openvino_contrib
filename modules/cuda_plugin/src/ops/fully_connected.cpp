// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected.hpp"

#include <cuda/blas.hpp>
#include <cuda_operation_registry.hpp>
#include <gsl/gsl_assert>
#include <ngraph/node.hpp>
#include <ngraph/op/matmul.hpp>
#include <utility>

#include "constant_factory.hpp"
#include "converters.hpp"
#include "matmul.hpp"

namespace CUDAPlugin {

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
    Expects(biasShape.size() > 0);
    MatMulOp::BroadcastToMatrix(biasShape);
    const auto biasShapeSize = ngraph::shape_size(biasShape);
    const auto matrixShapeSize = ngraph::shape_size(matrixShape);
    Expects(matrixShapeSize >= biasShapeSize);
    auto batchBiasCount = MatMulOp::GetMatrixNumBatches(biasShape);
    auto matMulBatchCount = matmul_op_.GetBatchCount();
    Expects(matMulBatchCount >= batchBiasCount);
    batch_bias_count_ = matrixShapeSize / biasShapeSize;
}

void FullyConnectedOp::Execute(const InferenceRequestContext& context,
                               Inputs inputs,
                               Outputs outputs,
                               const Workbuffers& workbuffers) const {
    Expects(inputs.size() == 3);
    Expects(outputs.size() == 1);
    auto& stream = context.getThreadContext().stream();

    auto bias = inputs[2];
    auto matrixC = outputs[0];
    for (size_t i = 0; i < batch_bias_count_; ++i) {
        stream.transfer(matrixC + i * bias_size_, bias, bias_size_);
    }
    matmul_op_.Execute(context, inputs.first(inputs.size() - 1), outputs, workbuffers);
}

OPERATION_REGISTER(FullyConnectedOp, FullyConnected);
}  // namespace CUDAPlugin
