// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <vector>

#include "kernel_ir/gfx_codegen_desc.hpp"
#include "mlir/gfx_apple_stage_pipeline.hpp"
#include "mlir/gfx_mpsrt_metadata.hpp"
#include "openvino/core/shape.hpp"
#include "runtime/gfx_stage_policy.hpp"

namespace ov {
namespace gfx_plugin {

namespace detail {

inline ov::element::Type gfx_mpsrt_matmul_resolve_type(const ov::element::Type& type,
                                                       const ov::element::Type& fallback) {
    if (type != ov::element::dynamic) {
        return type;
    }
    return fallback == ov::element::dynamic ? ov::element::f32 : fallback;
}

inline std::vector<int64_t> gfx_mpsrt_matmul_matrix_shape(int64_t batch,
                                                          int64_t rows,
                                                          int64_t columns) {
    if (batch == 1) {
        return {rows, columns};
    }
    return {batch, rows, columns};
}

inline std::vector<int64_t> gfx_mpsrt_matmul_input_matrix_shape(const ov::Shape& shape,
                                                                int64_t flattened_batch,
                                                                int64_t rows,
                                                                int64_t columns) {
    if (shape.size() >= 2) {
        std::vector<int64_t> dims;
        dims.reserve(shape.size());
        for (const auto dim : shape) {
            dims.push_back(static_cast<int64_t>(dim));
        }
        return dims;
    }
    return gfx_mpsrt_matmul_matrix_shape(flattened_batch, rows, columns);
}

}  // namespace detail

enum class GfxMatMulMpsrtLoweringKind {
    None,
    MpsGemm,
    MpsGemmWithMslEpilogue,
};

inline bool is_supported_mpsrt_matmul_epilogue_activation(ActivationKind kind) {
    switch (kind) {
        case ActivationKind::Relu:
        case ActivationKind::Sigmoid:
        case ActivationKind::Tanh:
        case ActivationKind::Gelu:
        case ActivationKind::Swish:
        case ActivationKind::HSwish:
        case ActivationKind::HSigmoid:
        case ActivationKind::Abs:
        case ActivationKind::Sign:
        case ActivationKind::Identity:
            return true;
        default:
            return false;
    }
}

inline bool can_lower_matmul_to_mpsrt_gemm(const MatMulCodegenDesc& desc) {
    return desc.batch > 0 &&
           (desc.batch_a == desc.batch || desc.batch_a == 1) &&
           (desc.batch_b == desc.batch || desc.batch_b == 1);
}

inline void annotate_module_with_matmul_mpsrt_epilogue_plan(mlir::ModuleOp module,
                                                            const MatMulCodegenDesc& desc,
                                                            const ov::Shape& shape_a,
                                                            const ov::Shape& shape_b) {
    if (!module) {
        return;
    }

    const auto output_type = detail::gfx_mpsrt_matmul_resolve_type(desc.output_type, desc.element_type);
    const auto bias_type = detail::gfx_mpsrt_matmul_resolve_type(desc.bias_type, output_type);
    const auto lhs_type = detail::gfx_mpsrt_matmul_resolve_type(desc.input_a_type, output_type);
    const auto rhs_type = detail::gfx_mpsrt_matmul_resolve_type(desc.input_b_type, output_type);
    const auto lhs_shape = detail::gfx_mpsrt_matmul_input_matrix_shape(shape_a,
                                                                       desc.batch_a,
                                                                       desc.a_transpose ? desc.K : desc.M,
                                                                       desc.a_transpose ? desc.M : desc.K);
    const auto rhs_shape = detail::gfx_mpsrt_matmul_input_matrix_shape(shape_b,
                                                                       desc.batch_b,
                                                                       desc.b_transpose ? desc.N : desc.K,
                                                                       desc.b_transpose ? desc.K : desc.N);
    const auto gemm_shape = detail::gfx_mpsrt_matmul_matrix_shape(desc.batch, desc.M, desc.N);
    const auto output_shape = gemm_shape;

    const auto lhs_desc = gfx_mpsrt_make_tensor_desc(lhs_shape,
                                                     lhs_type,
                                                     GfxStageStorageKind::Matrix,
                                                     GfxMpsrtTensorFlagExternalIo);
    const auto rhs_desc = gfx_mpsrt_make_tensor_desc(rhs_shape,
                                                     rhs_type,
                                                     GfxStageStorageKind::Matrix,
                                                     GfxMpsrtTensorFlagExternalIo);
    const auto gemm_desc = gfx_mpsrt_make_tensor_desc(gemm_shape,
                                                      output_type,
                                                      GfxStageStorageKind::Matrix);
    const auto output_desc = gfx_mpsrt_make_tensor_desc(output_shape,
                                                        output_type,
                                                        GfxStageStorageKind::Buffer,
                                                        GfxMpsrtTensorFlagExternalIo);

    GfxAppleMpsGemmMslEpilogueProgramDesc program_desc{};
    program_desc.lhs = lhs_desc;
    program_desc.rhs = rhs_desc;
    program_desc.gemm_output = gemm_desc;
    program_desc.output = output_desc;
    program_desc.gemm.transpose_lhs = desc.a_transpose ? 1u : 0u;
    program_desc.gemm.transpose_rhs = desc.b_transpose ? 1u : 0u;
    program_desc.gemm.alpha = 1.0f;
    program_desc.gemm.beta = 0.0f;
    program_desc.has_bias = desc.has_bias;
    if (desc.has_bias) {
        program_desc.bias = gfx_mpsrt_make_tensor_desc({desc.bias_dims[0],
                                                        desc.bias_dims[1],
                                                        desc.bias_dims[2]},
                                                       bias_type,
                                                       GfxStageStorageKind::Buffer,
                                                       GfxMpsrtTensorFlagExternalIo);
    }

    (void)materialize_apple_mps_gemm_msl_epilogue_program(module, program_desc);
}

inline GfxMatMulMpsrtLoweringKind annotate_module_with_matmul_mpsrt_plan(
    mlir::ModuleOp module,
    const GfxStageOptimizationPlan& plan,
    const MatMulCodegenDesc& desc,
    const ov::Shape& shape_a,
    const ov::Shape& shape_b) {
    if (!module || !can_lower_matmul_to_mpsrt_gemm(desc)) {
        return GfxMatMulMpsrtLoweringKind::None;
    }
    if (plan.placement.domain != GfxStageBackendDomain::AppleMps ||
        !plan.placement.uses_vendor_primitive) {
        return GfxMatMulMpsrtLoweringKind::None;
    }

    const bool needs_msl_epilogue = desc.has_bias || desc.has_activation;
    if (!needs_msl_epilogue) {
        const auto output_type = detail::gfx_mpsrt_matmul_resolve_type(desc.output_type,
                                                                       desc.element_type);
        const auto lhs_type = detail::gfx_mpsrt_matmul_resolve_type(desc.input_a_type,
                                                                    output_type);
        const auto rhs_type = detail::gfx_mpsrt_matmul_resolve_type(desc.input_b_type,
                                                                    output_type);
        const auto lhs_shape = detail::gfx_mpsrt_matmul_input_matrix_shape(
            shape_a,
            desc.batch_a,
            desc.a_transpose ? desc.K : desc.M,
            desc.a_transpose ? desc.M : desc.K);
        const auto rhs_shape = detail::gfx_mpsrt_matmul_input_matrix_shape(
            shape_b,
            desc.batch_b,
            desc.b_transpose ? desc.N : desc.K,
            desc.b_transpose ? desc.K : desc.N);
        const auto output_shape = detail::gfx_mpsrt_matmul_matrix_shape(desc.batch, desc.M, desc.N);

        GfxAppleMpsGemmProgramDesc program_desc{};
        program_desc.lhs = gfx_mpsrt_make_tensor_desc(lhs_shape,
                                                      lhs_type,
                                                      GfxStageStorageKind::Matrix,
                                                      GfxMpsrtTensorFlagExternalIo);
        program_desc.rhs = gfx_mpsrt_make_tensor_desc(rhs_shape,
                                                      rhs_type,
                                                      GfxStageStorageKind::Matrix,
                                                      GfxMpsrtTensorFlagExternalIo);
        program_desc.output = gfx_mpsrt_make_tensor_desc(output_shape,
                                                         output_type,
                                                         GfxStageStorageKind::Matrix,
                                                         GfxMpsrtTensorFlagExternalIo);
        program_desc.gemm.transpose_lhs = desc.a_transpose ? 1u : 0u;
        program_desc.gemm.transpose_rhs = desc.b_transpose ? 1u : 0u;
        program_desc.gemm.alpha = 1.0f;
        program_desc.gemm.beta = 0.0f;

        const auto materialized = materialize_apple_mps_gemm_program(module, program_desc);
        if (!materialized.valid || !materialized.typed_program_materialized) {
            return GfxMatMulMpsrtLoweringKind::None;
        }
        return GfxMatMulMpsrtLoweringKind::MpsGemm;
    }

    if (desc.has_activation && !is_supported_mpsrt_matmul_epilogue_activation(desc.activation)) {
        return GfxMatMulMpsrtLoweringKind::None;
    }

    annotate_module_with_matmul_mpsrt_epilogue_plan(module, desc, shape_a, shape_b);
    return GfxMatMulMpsrtLoweringKind::MpsGemmWithMslEpilogue;
}

}  // namespace gfx_plugin
}  // namespace ov
