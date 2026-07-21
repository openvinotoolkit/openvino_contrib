// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda/device_pointers.hpp>
#include <cuda_operation_base.hpp>
#include <transformer/nodes/fully_connected.hpp>

#include "cuda/constant_factory.hpp"
#include "openvino/op/matmul.hpp"

namespace ov {
namespace nvidia_gpu {

class MatMulOp : public OperationCuBlas {
public:
    using NodeOp = ov::op::v0::MatMul;

    template <typename TOperation>
    MatMulOp(const CreationContext& context,
             const TOperation& node,
             IndexCollection&& inputIds,
             IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibilityImpl() const override;

    int GetBatchCount() const { return batch_count_; }

    /**
     * Get number of batches that equals to product between dimensions in range [matrixShape.begin(),
     * matrixShape.end()-2)
     * @param matrixShape Matrix to calculate number of batches
     * @return Number of batches
     */
    static int GetMatrixNumBatches(const ov::Shape& matrixShape);
    /**
     * Broadcast some shape to Matrix
     * For example:
     * {} -> {1, 1}
     * {2} -> {1, 2}
     * {3, 2} -> {3, 2} // Not changed
     * @param shape Shape to broadcast to matrix
     */
    static void BroadcastToMatrix(ov::Shape& shape);

    /**
     * Get compute type according A/B matrix data type and C matrix data type
     * @param abDataType A/B matrix data type
     * @param cDataType C matrix data type
     * @return Available compute type
     */
    static cudaDataType_t GetComputeType(cudaDataType_t abDataType, cudaDataType_t cDataType);

private:
    /**
     * Broadcast input shapes according OpenVINO documentation:
     * @reference https://docs.openvinotoolkit.org/latest/openvino_docs_ops_matrix_MatMul_1.html
     * @param matrixAShape Shape of matrix A
     * @param matrixBShape Shape of matrix B
     * @param matrixCShape Shape of matrix C
     */
    static void BroadcastShapes(
        ov::Shape& matrixAShape, bool& transposeA, ov::Shape& matrixBShape, bool& transposeB, ov::Shape& matrixCShape);

    cudaDataType_t data_type_ = cudaDataType_t::CUDA_R_32F;
    cudaDataType_t compute_type_ = cudaDataType_t::CUDA_R_32F;
    int m_ = 0;
    int k_ = 0;
    int n_ = 0;
    int ld_a_ = 0;
    int ld_b_ = 0;
    int ld_c_ = 0;
    long long stride_a_ = 0;
    long long stride_b_ = 0;
    long long stride_c_ = 0;
    int batch_count_ = 0;
    const CUDA::constants::AnyNumeric* beta_ = nullptr;
    cublasOperation_t cublas_transpose_a_ = CUBLAS_OP_N;
    cublasOperation_t cublas_transpose_b_ = CUBLAS_OP_N;
};

}  // namespace nvidia_gpu
}  // namespace ov
