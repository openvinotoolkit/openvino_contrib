// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matmul.hpp"

#include <cuda/blas.hpp>
#include <cuda/float16.hpp>
#include <cuda_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/matmul.hpp>
#include <transformer/nodes/fully_connected.hpp>
#include <utility>

#include "converters.hpp"
#include "cuda/constant_factory.hpp"

namespace ov {
namespace nvidia_gpu {

template <typename TOperation>
MatMulOp::MatMulOp(const CreationContext& context,
                   const TOperation& op,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds)
    : OperationCuBlas(context, op, std::move(inputIds), std::move(outputIds)) {
    OPENVINO_ASSERT(op.get_input_size() >= 2, "Node name: ", GetName());
    OPENVINO_ASSERT(op.get_output_size() == 1, "Node name: ", GetName());
    OPENVINO_ASSERT(convertDataType<cudaDataType_t>(op.get_input_element_type(0)) ==
                        convertDataType<cudaDataType_t>(op.get_input_element_type(1)),
                    "Node name: ",
                    GetName());
    data_type_ = convertDataType<cudaDataType_t>(op.get_input_element_type(0));
    compute_type_ = GetComputeType(data_type_, convertDataType<cudaDataType_t>(op.get_output_element_type(0)));
    auto inputAShape = op.get_input_shape(0);
    auto inputBShape = op.get_input_shape(1);
    auto outputCShape = op.get_output_shape(0);
    OPENVINO_ASSERT(inputAShape.size() > 0, "Node name: ", GetName());
    OPENVINO_ASSERT(inputBShape.size() > 0, "Node name: ", GetName());
    bool transposeA = op.get_transpose_a();
    bool transposeB = op.get_transpose_b();
    const int batchACount = GetMatrixNumBatches(inputAShape);
    const int batchBCount = GetMatrixNumBatches(inputBShape);
    BroadcastShapes(inputAShape, transposeA, inputBShape, transposeB, outputCShape);
    batch_count_ = std::max(batchACount, batchBCount);
    const size_t rowsA = *(inputAShape.end() - !transposeA - 1);
    const size_t colsA = *(inputAShape.end() - transposeA - 1);
    const size_t rowsB = *(inputBShape.end() - !transposeB - 1);
    const size_t colsB = *(inputBShape.end() - transposeB - 1);
    OPENVINO_ASSERT(colsA == rowsB, "Node name: ", GetName());
    m_ = rowsA;
    k_ = colsA;
    n_ = colsB;
    ld_a_ = *(inputAShape.end() - 1);
    ld_b_ = *(inputBShape.end() - 1);
    ld_c_ = *(outputCShape.end() - 1);
    stride_a_ = (batchACount > 1) ? (m_ * k_) : 0;
    stride_b_ = (batchBCount > 1) ? (k_ * n_) : 0;
    stride_c_ = (m_ * n_);
    cublas_transpose_a_ = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublas_transpose_b_ = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;
    if constexpr (std::is_same_v<TOperation, nodes::FullyConnected>) {
        beta_ = &CUDA::NumericConst<CUDA::constants::one>(compute_type_);
    } else {
        beta_ = &CUDA::NumericConst<CUDA::constants::zero>(compute_type_);
    }
    OPENVINO_ASSERT(m_ != 0, "Node name: ", GetName());
    OPENVINO_ASSERT(k_ != 0, "Node name: ", GetName());
    OPENVINO_ASSERT(n_ != 0, "Node name: ", GetName());
    OPENVINO_ASSERT(ld_a_ != 0, "Node name: ", GetName());
    OPENVINO_ASSERT(ld_b_ != 0, "Node name: ", GetName());
    OPENVINO_ASSERT(ld_c_ != 0, "Node name: ", GetName());
    OPENVINO_ASSERT(batch_count_ != 0, "Node name: ", GetName());
}
template MatMulOp::MatMulOp(const CreationContext& context,
                            const ov::op::v0::MatMul&,
                            IndexCollection&&,
                            IndexCollection&&);
template MatMulOp::MatMulOp(const CreationContext& context,
                            const nodes::FullyConnected&,
                            IndexCollection&&,
                            IndexCollection&&);

cudaDataType_t MatMulOp::GetComputeType(const cudaDataType_t abDataType, const cudaDataType_t cDataType) {
    constexpr auto SwitchCase = [](cudaDataType_t a, cudaDataType_t b) constexpr { return (a << 16) + b; };
    /**
     * NOTE: This switch is an implementation of CuBlas table for available compute types:
     * @reference https://docs.nvidia.com/cuda/cublas/index.html#cublas-GemmStridedBatchedEx
     */
    switch (SwitchCase(abDataType, cDataType)) {
        case SwitchCase(CUDA_R_16F, CUDA_R_16F): {
            return CUDA_R_16F;
        }
        case SwitchCase(CUDA_R_8I, CUDA_R_32I): {
            return CUDA_R_32I;
        }
#ifdef CUDA_HAS_BF16_TYPE
        case SwitchCase(CUDA_R_16BF, CUDA_R_16BF):
        case SwitchCase(CUDA_R_16BF, CUDA_R_32F):
#endif
        case SwitchCase(CUDA_R_8I, CUDA_R_32F):
        case SwitchCase(CUDA_R_16F, CUDA_R_32F):
        case SwitchCase(CUDA_R_32F, CUDA_R_32F):
        case SwitchCase(CUDA_C_8I, CUDA_C_32F):
        case SwitchCase(CUDA_C_32F, CUDA_C_32F): {
            return CUDA_R_32F;
        }
        case SwitchCase(CUDA_R_64F, CUDA_R_64F):
        case SwitchCase(CUDA_C_64F, CUDA_C_64F): {
            return CUDA_R_64F;
        }
        default:
            throw_ov_exception(
                fmt::format("Not supported combination of A and B types [{}] "
                            "with C type [{}]",
                            abDataType,
                            cDataType));
    }
}

int MatMulOp::GetMatrixNumBatches(const ov::Shape& matrixShape) {
    return matrixShape.size() >= 2
               ? std::accumulate(matrixShape.begin(), matrixShape.end() - 2, 1, std::multiplies<size_t>())
               : 1;
}

void MatMulOp::BroadcastShapes(
    ov::Shape& matrixAShape, bool& transposeA, ov::Shape& matrixBShape, bool& transposeB, ov::Shape& matrixCShape) {
    /**
     * NOTE: See NGraph documentation for broadcasting:
     * @reference https://docs.openvinotoolkit.org/latest/openvino_docs_ops_matrix_MatMul_1.html
     */
    if (matrixAShape.size() == 1 && matrixBShape.size() == 1) {
        // 1D x 1D: [X] x [X] -> [1, X] x [X, 1] -> [1, 1] => [] (scalar)
        matrixAShape = ov::Shape{1, matrixAShape[0]};
        matrixBShape = ov::Shape{matrixBShape[0], 1};
        transposeA = false;
        transposeB = false;
    } else if (matrixAShape.size() == 1 && matrixBShape.size() > 1) {
        // 1D x ND: [X] x [B, ..., X, Y] -> [1, X] x [B, ..., X, Y] -> [B, ..., 1, Y] => [B, ..., Y]
        matrixAShape = ov::Shape{1, matrixAShape[0]};
        transposeA = false;
    } else if (matrixAShape.size() > 1 && matrixBShape.size() == 1) {
        // ND x 1D: [B, ..., X, Y] x [Y] -> [B, ..., X, Y] x [Y, 1] -> [B, ..., X, 1] => [B, ..., X]
        matrixBShape = ov::Shape{matrixBShape[0], 1};
        transposeB = false;
    } else if (matrixAShape.size() > 1 && matrixBShape.size() > 1) {
        // ND x ND: [B, ..., X, Y] x [B, ..., Y, Z] => [B, ..., X, Z]
        auto broadcastNdToMd = [](const auto& shapeToBroadcast, auto& broadcastShape) {
            OPENVINO_ASSERT(shapeToBroadcast.size() >= broadcastShape.size());
            std::vector<size_t> newAxies;
            newAxies.reserve(shapeToBroadcast.size());
            newAxies.insert(newAxies.end(), shapeToBroadcast.begin(), shapeToBroadcast.end() - 2);
            newAxies.insert(newAxies.end(), broadcastShape.end() - 2, broadcastShape.end());
            broadcastShape = ov::Shape{newAxies};
        };
        const size_t batchA = GetMatrixNumBatches(matrixAShape);
        const size_t batchB = GetMatrixNumBatches(matrixBShape);
        if (batchA > batchB) {
            broadcastNdToMd(matrixAShape, matrixBShape);
        } else if (batchA < batchB) {
            broadcastNdToMd(matrixBShape, matrixAShape);
        }
        OPENVINO_ASSERT(GetMatrixNumBatches(matrixAShape) == GetMatrixNumBatches(matrixBShape));
    }
    OPENVINO_ASSERT(*(matrixAShape.end() - transposeA - 1) == *(matrixBShape.end() - !transposeB - 1));
    if (matrixAShape.size() > matrixBShape.size()) {
        matrixCShape = matrixAShape;
    } else {
        matrixCShape = matrixBShape;
    }
    *(matrixCShape.end() - 2) = *(matrixAShape.end() - !transposeA - 1);
    *(matrixCShape.end() - 1) = *(matrixBShape.end() - transposeB - 1);
}

void MatMulOp::BroadcastToMatrix(ov::Shape& shape) {
    if (shape.size() < 2) {
        shape.insert(shape.begin(), 2 - shape.size(), 1);
    }
}

// NOTE: Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n), C is stored as row-major matrix
void MatMulOp::Execute(const InferenceRequestContext& context,
                       Inputs inputs,
                       Outputs outputs,
                       const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() == 2, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());
    auto& cuBlasHandle = context.getThreadContext().cuBlasHandle();
    auto matrixA = inputs[0];
    auto matrixB = inputs[1];
    auto matrixC = outputs[0];

    /**
     * NOTE: A and B are switched in places. A returns k as leading dimension and B returns n.
     *       Such workaround is done, because cuBlas works with column-major matrices,
     *       but we need to get output row-major matrix
     *       Instead of computing C = A x B (cuBlas will return in column-major format),
     *       we compute Ct = Bt x At (where t means transposed)
     *       As result Ct would be row-major matrix
     */
    throwIfError(cublasGemmStridedBatchedEx(cuBlasHandle.get(),
                                            cublas_transpose_b_,
                                            cublas_transpose_a_,
                                            n_,
                                            m_,
                                            k_,
                                            &CUDA::NumericConst<CUDA::constants::one>(compute_type_),
                                            matrixB.get(),
                                            data_type_,
                                            ld_b_,
                                            stride_b_,
                                            matrixA.get(),
                                            data_type_,
                                            ld_a_,
                                            stride_a_,
                                            beta_,
                                            matrixC.get(),
                                            data_type_,
                                            ld_c_,
                                            stride_c_,
                                            batch_count_,
                                            compute_type_,
                                            CUBLAS_GEMM_DEFAULT));
}

CudaGraphCompatibility MatMulOp::GetCudaGraphCompatibility() const { return CudaGraphCompatibility::FULL; }

OPERATION_REGISTER(MatMulOp, MatMul);
}  // namespace nvidia_gpu
}  // namespace ov
