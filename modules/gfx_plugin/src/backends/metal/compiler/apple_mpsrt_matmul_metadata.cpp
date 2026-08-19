// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/compiler/apple_mpsrt_matmul_metadata.hpp"

#include <utility>
#include <vector>

#include "backends/metal/compiler/apple_stage_pipeline.hpp"
#include "backends/metal/compiler/apple_vendor_descriptors.hpp"
#include "mlir/gfx_backend_custom_kernel_adapter.hpp"
#include "mlir/gfx_mpsrt_metadata.hpp"
#include "openvino/core/except.hpp"
#include "runtime/gfx_compile_profiling.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

ov::element::Type resolve_matmul_type(const ov::element::Type& type,
                                      const ov::element::Type& fallback) {
    if (type != ov::element::dynamic) {
        return type;
    }
    return fallback == ov::element::dynamic ? ov::element::f32 : fallback;
}

std::vector<int64_t> matmul_matrix_shape(int64_t batch, int64_t rows, int64_t columns) {
    if (batch == 1) {
        return {rows, columns};
    }
    return {batch, rows, columns};
}

std::vector<int64_t> matmul_input_matrix_shape(const ov::Shape& shape,
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
    return matmul_matrix_shape(flattened_batch, rows, columns);
}

GfxMpsrtGemmAbiDesc make_mpsrt_gemm_abi_desc(const MatMulCodegenDesc& desc) {
    GfxMpsrtGemmAbiDesc gemm{};
    gemm.transpose_lhs = desc.a_transpose ? 1u : 0u;
    gemm.transpose_rhs = desc.b_transpose ? 1u : 0u;
    gemm.alpha = 1.0f;
    gemm.beta = 0.0f;
    return gemm;
}

GfxMpsrtStageDesc make_mps_gemm_stage_desc(const GfxMpsrtGemmAbiDesc& gemm) {
    GfxMpsrtStageDesc stage{};
    stage.kind = GfxMpsrtStageKind::MPSGemm;
    stage.domain = GfxStageBackendDomain::AppleMps;
    stage.input_storage = GfxMpsrtStorage::Matrix;
    stage.output_storage = GfxMpsrtStorage::Matrix;
    stage.layout = GfxMpsrtLayout::RowMajor;
    stage.kernel_name = "mps_gemm";
    stage.gemm_desc = gemm;
    stage.gemm_desc.alpha = stage.gemm_desc.alpha == 0.0f ? 1.0f : stage.gemm_desc.alpha;
    stage.stage_manifest = make_gfx_vendor_stage_manifest(GfxKernelStageFamily::Gemm,
                                                          GfxKernelBackendDomain::AppleMps,
                                                          GfxKernelStorageKind::Matrix,
                                                          "apple_mps:matrix:MatMul");
    return stage;
}

GfxMpsrtStageDesc make_msl_gemm_epilogue_stage_desc(bool has_bias) {
    const auto binding = make_backend_custom_kernel_roles_binding_plan(
        "MatMulEpilogue",
        "eltwise_fused_buffer",
        has_bias
            ? std::vector<GfxKernelBufferRole>{GfxKernelBufferRole::TensorInput,
                                               GfxKernelBufferRole::TensorInput,
                                               GfxKernelBufferRole::TensorOutput}
            : std::vector<GfxKernelBufferRole>{GfxKernelBufferRole::TensorInput,
                                               GfxKernelBufferRole::TensorOutput},
        GfxKernelBackendDomain::AppleMsl);

    GfxMpsrtStageDesc stage{};
    stage.kind = GfxMpsrtStageKind::MSLDispatch;
    stage.domain = GfxStageBackendDomain::AppleMsl;
    stage.input_storage = GfxMpsrtStorage::Buffer;
    stage.output_storage = GfxMpsrtStorage::Buffer;
    stage.layout = GfxMpsrtLayout::Linear;
    stage.kernel_name = "eltwise_fused_buffer";
    OPENVINO_ASSERT(binding.valid,
                    "GFX MPSRT MatMul: failed to materialize MSL epilogue custom-kernel manifest");
    stage.stage_manifest = binding.stage_manifest;
    return stage;
}

}  // namespace

bool is_supported_mpsrt_matmul_epilogue_activation(ActivationKind kind) {
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

bool can_lower_matmul_to_mpsrt_gemm(const MatMulCodegenDesc& desc) {
    return desc.batch > 0 &&
           (desc.batch_a == desc.batch || desc.batch_a == 1) &&
           (desc.batch_b == desc.batch || desc.batch_b == 1);
}

void annotate_module_with_matmul_mpsrt_epilogue_plan(mlir::ModuleOp module,
                                                     const MatMulCodegenDesc& desc,
                                                     const ov::Shape& shape_a,
                                                     const ov::Shape& shape_b) {
    if (!module) {
        return;
    }

    const auto output_type = resolve_matmul_type(desc.output_type, desc.element_type);
    const auto bias_type = resolve_matmul_type(desc.bias_type, output_type);
    const auto lhs_type = resolve_matmul_type(desc.input_a_type, output_type);
    const auto rhs_type = resolve_matmul_type(desc.input_b_type, output_type);
    const auto lhs_shape = matmul_input_matrix_shape(shape_a,
                                                     desc.batch_a,
                                                     desc.a_transpose ? desc.K : desc.M,
                                                     desc.a_transpose ? desc.M : desc.K);
    const auto rhs_shape = matmul_input_matrix_shape(shape_b,
                                                     desc.batch_b,
                                                     desc.b_transpose ? desc.N : desc.K,
                                                     desc.b_transpose ? desc.K : desc.N);
    const auto gemm_shape = matmul_matrix_shape(desc.batch, desc.M, desc.N);
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

    GfxAppleMpsrtProgramPlanBuilder builder("mps_gemm_plus_msl_epilogue_model|MatMul");
    const auto lhs_value = builder.add_external_input(lhs_desc);
    const auto rhs_value = builder.add_external_input(rhs_desc);

    GfxMpsrtValue bias_value = 0u;
    if (desc.has_bias) {
        bias_value = builder.add_external_input(
            gfx_mpsrt_make_tensor_desc({desc.bias_dims[0],
                                        desc.bias_dims[1],
                                        desc.bias_dims[2]},
                                       bias_type,
                                       GfxStageStorageKind::Buffer,
                                       GfxMpsrtTensorFlagExternalIo));
    }

    const auto gemm_output_value = builder.add_single_output_stage(
        make_mps_gemm_stage_desc(make_mpsrt_gemm_abi_desc(desc)),
        {lhs_value, rhs_value},
        gemm_desc);

    std::vector<GfxMpsrtValue> epilogue_inputs = {gemm_output_value};
    if (desc.has_bias) {
        epilogue_inputs.push_back(bias_value);
    }
    const auto output_value = builder.add_single_output_stage(
        make_msl_gemm_epilogue_stage_desc(desc.has_bias),
        std::move(epilogue_inputs),
        output_desc);
    builder.add_external_output(output_value);

    const auto program_plan = builder.finalize();
    (void)materialize_apple_mpsrt_program_plan(module, program_plan);
}

GfxMatMulMpsrtLoweringKind annotate_module_with_matmul_mpsrt_plan(
    mlir::ModuleOp module,
    const GfxStageOptimizationPlan& plan,
    const MatMulCodegenDesc& desc,
    const ov::Shape& shape_a,
    const ov::Shape& shape_b) {
    if (!module) {
        increment_compile_counter("matmul_mpsrt_reject_no_module_count");
        return GfxMatMulMpsrtLoweringKind::None;
    }
    if (!can_lower_matmul_to_mpsrt_gemm(desc)) {
        increment_compile_counter("matmul_mpsrt_reject_batch_broadcast_count");
        return GfxMatMulMpsrtLoweringKind::None;
    }
    if (plan.placement.domain != GfxStageBackendDomain::AppleMps ||
        !plan.placement.uses_vendor_primitive) {
        increment_compile_counter("matmul_mpsrt_reject_non_mps_placement_count");
        return GfxMatMulMpsrtLoweringKind::None;
    }

    const bool needs_msl_epilogue = desc.has_bias || desc.has_activation;
    if (!needs_msl_epilogue) {
        const auto output_type = resolve_matmul_type(desc.output_type, desc.element_type);
        const auto lhs_type = resolve_matmul_type(desc.input_a_type, output_type);
        const auto rhs_type = resolve_matmul_type(desc.input_b_type, output_type);
        const auto lhs_shape = matmul_input_matrix_shape(
            shape_a,
            desc.batch_a,
            desc.a_transpose ? desc.K : desc.M,
            desc.a_transpose ? desc.M : desc.K);
        const auto rhs_shape = matmul_input_matrix_shape(
            shape_b,
            desc.batch_b,
            desc.b_transpose ? desc.N : desc.K,
            desc.b_transpose ? desc.K : desc.N);
        const auto output_shape = matmul_matrix_shape(desc.batch, desc.M, desc.N);

        GfxAppleMpsVendorPrimitiveContract contract{};
        if (!gfx_apple_make_mps_gemm_contract(
                make_mpsrt_gemm_abi_desc(desc),
                gfx_mpsrt_make_tensor_desc(lhs_shape,
                                           lhs_type,
                                           GfxStageStorageKind::Matrix,
                                           GfxMpsrtTensorFlagExternalIo),
                gfx_mpsrt_make_tensor_desc(rhs_shape,
                                           rhs_type,
                                           GfxStageStorageKind::Matrix,
                                           GfxMpsrtTensorFlagExternalIo),
                gfx_mpsrt_make_tensor_desc(output_shape,
                                           output_type,
                                           GfxStageStorageKind::Matrix,
                                           GfxMpsrtTensorFlagExternalIo),
                contract)) {
            increment_compile_counter("matmul_mpsrt_reject_contract_count");
            return GfxMatMulMpsrtLoweringKind::None;
        }

        const auto materialized =
            materialize_apple_mps_vendor_contract_program(module, plan, "MatMul", contract);
        if (!materialized.valid || !materialized.typed_program_materialized) {
            increment_compile_counter("matmul_mpsrt_reject_materialization_count");
            return GfxMatMulMpsrtLoweringKind::None;
        }
        increment_compile_counter("matmul_mpsrt_accept_mps_gemm_count");
        return GfxMatMulMpsrtLoweringKind::MpsGemm;
    }

    if (desc.has_activation && !is_supported_mpsrt_matmul_epilogue_activation(desc.activation)) {
        increment_compile_counter("matmul_mpsrt_reject_unsupported_epilogue_activation_count");
        return GfxMatMulMpsrtLoweringKind::None;
    }

    annotate_module_with_matmul_mpsrt_epilogue_plan(module, desc, shape_a, shape_b);
    increment_compile_counter("matmul_mpsrt_accept_mps_gemm_msl_epilogue_count");
    return GfxMatMulMpsrtLoweringKind::MpsGemmWithMslEpilogue;
}

}  // namespace gfx_plugin
}  // namespace ov
