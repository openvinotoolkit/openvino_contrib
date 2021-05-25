// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include <gpu/device_pointers.hpp>

namespace CUDAPlugin {

class MatMulOp : public OperationBase {
public:
    MatMulOp(const std::shared_ptr<ngraph::Node>& node,
             std::vector<unsigned>&& inputIds,
             std::vector<unsigned>&& outputIds);
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors) override;

private:
    /**
     * Get compute type according A/B matrix data type and C matrix data type
     * @param abDataType A/B matrix data type
     * @param cDataType C matrix data type
     * @return Available compute type
     */
    static cudaDataType_t GetComputeType(cudaDataType_t abDataType, cudaDataType_t cDataType);
    /**
     * Get scaling alpha constant according compute type
     * @param computeType Data type for getting constant
     * @return Pointer to constant
     */
    static const void* ScalingAlphaConstant(cudaDataType_t computeType);
    /**
     * Get scaling beta constant according compute type
     * @param computeType Data type for getting constant
     * @return Pointer to constant
     */
    static const void* ScalingBetaConstant(cudaDataType_t computeType);
    /**
     * Get number of batches that equals to product between dimensions in range [matrixShape.begin(), matrixShape.end()-2)
     * @param matrixShape Matrix to calculate number of batches
     * @return Number of batches
     */
    static int GetNumBatches(const ngraph::Shape& matrixShape);
    /**
     * Broadcast input shapes according OpenVINO documentation:
     * @reference https://docs.openvinotoolkit.org/latest/openvino_docs_ops_matrix_MatMul_1.html
     * @param matrixAShape Shape of matrix A
     * @param matrixBShape Shape of matrix B
     * @param matrixCShape Shape of matrix C
     */
    static void BroadcastShapes(ngraph::Shape& matrixAShape,
                                bool& transposeA,
                                ngraph::Shape& matrixBShape,
                                bool& transposeB,
                                ngraph::Shape& matrixCShape);

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
    cublasOperation_t cublas_transpose_a_ = CUBLAS_OP_N;
    cublasOperation_t cublas_transpose_b_ = CUBLAS_OP_N;
    const void* alpha_ = nullptr;
    const void* beta_ = nullptr;
};

} // namespace CUDAPlugin
