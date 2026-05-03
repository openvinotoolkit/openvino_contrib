// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <vector>

#include "kernel_ir/gfx_codegen_desc.hpp"
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

inline GfxMpsrtStageDesc gfx_mpsrt_make_matmul_gemm_stage_desc(const MatMulCodegenDesc& desc) {
    GfxMpsrtStageDesc stage{};
    stage.kind = GfxMpsrtStageKind::MPSGemm;
    stage.domain = GfxStageBackendDomain::AppleMps;
    stage.input_storage = GfxMpsrtStorage::Matrix;
    stage.output_storage = GfxMpsrtStorage::Matrix;
    stage.layout = GfxMpsrtLayout::RowMajor;
    stage.uses_vendor_primitive = true;
    stage.stage_type = "MatMul";
    stage.kernel_name = "mps_gemm";
    stage.builder_symbol = gfx_mpsrt_builder_symbol(stage.kind);
    stage.specialization_key = "apple_mps:matrix:MatMul";
    stage.gemm_desc.transpose_lhs = desc.a_transpose ? 1u : 0u;
    stage.gemm_desc.transpose_rhs = desc.b_transpose ? 1u : 0u;
    stage.gemm_desc.alpha = 1.0f;
    stage.gemm_desc.beta = 0.0f;
    stage.stage_manifest = make_gfx_vendor_stage_manifest(GfxKernelStageFamily::Gemm,
                                                          GfxKernelBackendDomain::AppleMps,
                                                          GfxKernelStorageKind::Matrix,
                                                          stage.specialization_key);
    return stage;
}

inline GfxMpsrtStageDesc gfx_mpsrt_make_matmul_epilogue_stage_desc() {
    GfxMpsrtStageDesc stage{};
    stage.kind = GfxMpsrtStageKind::MSLDispatch;
    stage.domain = GfxStageBackendDomain::AppleMsl;
    stage.input_storage = GfxMpsrtStorage::Buffer;
    stage.output_storage = GfxMpsrtStorage::Buffer;
    stage.layout = GfxMpsrtLayout::Linear;
    stage.uses_custom_kernel = true;
    stage.stage_type = "MatMulEpilogue";
    stage.kernel_name = "eltwise_fused_buffer";
    stage.builder_symbol = gfx_mpsrt_builder_symbol(stage.kind);
    stage.specialization_key = "apple_msl:buffer:MatMulEpilogue";
    stage.stage_manifest = make_gfx_custom_kernel_stage_manifest(
        GfxKernelStageFamily::Eltwise,
        GfxKernelBackendDomain::AppleMsl,
        GfxKernelStorageKind::Buffer,
        stage.specialization_key,
        make_gfx_custom_kernel_manifest("eltwise_fused_buffer",
                                        static_cast<uint32_t>(GfxMslKernelFamily::EltwiseFusedBuffer),
                                        "eltwise_fused_buffer",
                                        make_gfx_kernel_tail_outputs_abi(),
                                        256,
                                        /*precompiled_binary_required=*/true));
    return stage;
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

    constexpr GfxMpsrtValue kLhsValue = 0;
    constexpr GfxMpsrtValue kRhsValue = 1;
    constexpr GfxMpsrtValue kBiasValue = 2;
    const GfxMpsrtValue gemm_value = desc.has_bias ? 3 : 2;
    const GfxMpsrtValue output_value = desc.has_bias ? 4 : 3;

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

    std::vector<GfxMpsrtTensorDesc> input_descs = {lhs_desc, rhs_desc};
    if (desc.has_bias) {
        input_descs.push_back(gfx_mpsrt_make_tensor_desc({desc.bias_dims[0],
                                                          desc.bias_dims[1],
                                                          desc.bias_dims[2]},
                                                         bias_type,
                                                         GfxStageStorageKind::Buffer,
                                                         GfxMpsrtTensorFlagExternalIo));
    }

    const auto gemm_stage_desc = detail::gfx_mpsrt_make_matmul_gemm_stage_desc(desc);
    const auto epilogue_stage_desc = detail::gfx_mpsrt_make_matmul_epilogue_stage_desc();
    const std::vector<GfxMpsrtValue> epilogue_inputs = desc.has_bias
                                                           ? std::vector<GfxMpsrtValue>{gemm_value, kBiasValue}
                                                           : std::vector<GfxMpsrtValue>{gemm_value};
    std::vector<GfxMpsrtExternalBufferRole> external_roles(input_descs.size(),
                                                           GfxMpsrtExternalBufferRole::TensorInput);
    external_roles.push_back(GfxMpsrtExternalBufferRole::TensorOutput);

    annotate_module_with_mpsrt_multi_stage_plan(
        module,
        "mps_gemm_plus_msl_epilogue_model|MatMul",
        input_descs,
        {GfxMpsrtBuilderStageSpec{gemm_stage_desc,
                                  gfx_mpsrt_stage_record_key(gemm_stage_desc),
                                  {kLhsValue, kRhsValue},
                                  {gemm_value},
                                  {gemm_desc}},
         GfxMpsrtBuilderStageSpec{epilogue_stage_desc,
                                  gfx_mpsrt_stage_record_key(epilogue_stage_desc),
                                  epilogue_inputs,
                                  {output_value},
                                  {output_desc}}},
        {output_value},
        external_roles);
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
        annotate_module_with_mpsrt_stage_plan(module, plan, "MatMul");
        GfxMpsrtGemmAbiDesc gemm_desc{};
        gemm_desc.transpose_lhs = desc.a_transpose ? 1u : 0u;
        gemm_desc.transpose_rhs = desc.b_transpose ? 1u : 0u;
        annotate_module_with_mpsrt_gemm_desc(module, gemm_desc);
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
