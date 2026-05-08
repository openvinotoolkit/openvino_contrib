// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen_matmul_metal.hpp"

#include "kernel_ir/gfx_kernel_manifest.hpp"
#include "mlir/codegen_common.hpp"
#include "mlir/gfx_mlir_kernel_builder.hpp"
#include "mlir/gfx_mpsrt_metadata.hpp"
#include "mlir/mlir_kernel_plan_utils.hpp"
#include "mlir/msl_codegen.hpp"
#include "mlir/msl_codegen_matmul_mpsrt.hpp"
#include "runtime/gfx_stage_policy.hpp"
#include "transforms/mlir_fused_ops.hpp"

#include "openvino/core/except.hpp"

#include <utility>

namespace ov {
namespace gfx_plugin {
namespace {

enum class GfxMatMulMetalKernelSourcePlanKind {
    None,
    Mpsrt,
    MslFallback,
};

struct GfxMatMulMetalKernelSourcePlan {
    GfxMatMulMetalKernelSourcePlanKind kind = GfxMatMulMetalKernelSourcePlanKind::None;
    GfxMatMulMpsrtLoweringKind mpsrt_lowering = GfxMatMulMpsrtLoweringKind::None;
    GfxMpsrtKernelSourcePlan mpsrt_plan;
    KernelSource source;
    bool requires_mpsrt_model = false;

    bool valid() const {
        return kind != GfxMatMulMetalKernelSourcePlanKind::None && source.module;
    }
};

ov::element::Type resolve_matmul_buffer_type(const ov::element::Type& type,
                                             const ov::element::Type& fallback) {
    if (type != ov::element::dynamic) {
        return type;
    }
    return fallback == ov::element::dynamic ? ov::element::f32 : fallback;
}

GfxKernelExternalBufferAbiSpec make_matmul_bias_external_buffer_abi() {
    return make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorInput,
                                      GfxKernelBufferRole::TensorInput,
                                      GfxKernelBufferRole::TensorInput,
                                      GfxKernelBufferRole::TensorOutput});
}

uint32_t resolve_matmul_source_arg_count(mlir::ModuleOp module,
                                         uint32_t arg_count,
                                         const KernelArgMappingInfo& info) {
    const uint32_t inferred_total =
        static_cast<uint32_t>(infer_kernel_arg_count_from_module(module,
                                                                 info.signature.total(),
                                                                 /*entry_point=*/{},
                                                                 /*allow_legacy_operand_attrs=*/false));
    if (arg_count != 0) {
        return arg_count;
    }
    return inferred_total;
}

GfxMatMulMetalKernelSourcePlan make_matmul_msl_fallback_source_plan(
    mlir::ModuleOp module,
    const GpuBufferManager* buffer_manager,
    const std::shared_ptr<const ov::Node>& node,
    const MatMulCodegenDesc& desc) {
    GfxMatMulMetalKernelSourcePlan result{};
    if (!module || !node) {
        return result;
    }

    constexpr const char* kStageType = "MatMul";
    constexpr const char* kEntryPoint = "matmul_kernel";
    const uint32_t arg_count = desc.has_bias ? 4u : 3u;
    auto plan = select_stage_optimization_plan(buffer_manager,
                                               GpuBackend::Metal,
                                               kStageType,
                                               node,
                                               desc.output_type,
                                               desc.has_bias,
                                               desc.has_activation,
                                               /*has_batchnorm=*/false,
                                               GfxStageRuntimeTraits{});
    force_apple_msl_buffer_placement(plan, kStageType);
    annotate_msl_module_with_stage_plan(module, plan, kStageType, kEntryPoint);
    if (desc.has_bias) {
        GfxKernelStageManifest manifest{};
        if (detail::gfx_mpsrt_read_stage_manifest_attrs(module, manifest) &&
            manifest.custom_kernel.valid) {
            manifest.custom_kernel.external_buffer_abi = make_matmul_bias_external_buffer_abi();
            detail::gfx_mpsrt_set_stage_manifest_attrs(module, manifest);
        }
    }

    auto plan_ctx = build_mlir_kernel_plan(
        module,
        kEntryPoint,
        node,
        /*output_args_override=*/0,
        /*extra_inputs=*/0,
        node->get_friendly_name().c_str(),
        "gfx_kernel",
        [&](const KernelArgMappingInfo& info) -> size_t {
            return resolve_matmul_source_arg_count(module, arg_count, info);
        });
    plan_ctx.build_info.plan.set_legacy_operand_attrs_policy(
        LegacyOperandAttrsPolicy::Reject);

    auto source_desc = desc;
    auto source = plan_ctx.build_info.plan.to_source_with_msl_generator(
        [source_desc](mlir::ModuleOp mod) {
            return generate_msl_from_mlir(mod, source_desc);
        });
    source.signature.output_arg_count = 1;

    result.kind = GfxMatMulMetalKernelSourcePlanKind::MslFallback;
    result.requires_mpsrt_model = false;
    auto mpsrt_plan = configure_msl_kernel_source_plan(std::move(source), kStageType);
    if (mpsrt_plan.valid()) {
        result.mpsrt_plan = std::move(mpsrt_plan);
        result.source = result.mpsrt_plan.source;
        result.requires_mpsrt_model = result.mpsrt_plan.requires_mpsrt_model;
    } else {
        result.kind = GfxMatMulMetalKernelSourcePlanKind::None;
    }
    return result;
}

GfxMatMulMetalKernelSourcePlan lower_matmul_node_to_metal_kernel_source(
    mlir::MLIRContext& ctx,
    const GpuBufferManager* buffer_manager,
    const std::shared_ptr<const ov::Node>& node,
    MatMulCodegenDesc desc,
    const ov::Shape& shape_a,
    const ov::Shape& shape_b) {
    GfxMatMulMetalKernelSourcePlan result{};
    if (!node) {
        return result;
    }

    auto module = build_mlir_for_node(node, ctx);
    if (!module) {
        return result;
    }
    if (desc.has_activation) {
        const bool applied = apply_fused_activation(module, desc.activation, desc.alpha);
        if (!applied) {
            return result;
        }
    }

    const auto output_type = node->get_output_element_type(0);
    desc.element_type = resolve_matmul_buffer_type(desc.element_type, output_type);
    desc.input_a_type = resolve_matmul_buffer_type(desc.input_a_type, desc.element_type);
    desc.input_b_type = resolve_matmul_buffer_type(desc.input_b_type, desc.element_type);
    desc.output_type = output_type;

    const auto placement = select_stage_optimization_plan(buffer_manager,
                                                          GpuBackend::Metal,
                                                          "MatMul",
                                                          node,
                                                          desc.output_type,
                                                          desc.has_bias,
                                                          desc.has_activation,
                                                          /*has_batchnorm=*/false,
                                                          GfxStageRuntimeTraits{});
    auto mpsrt_source = lower_matmul_module_to_mpsrt_kernel_source(module,
                                                                  placement,
                                                                  desc,
                                                                  shape_a,
                                                                  shape_b);
    if (mpsrt_source.valid()) {
        result.kind = GfxMatMulMetalKernelSourcePlanKind::Mpsrt;
        result.mpsrt_lowering = mpsrt_source.lowering;
        result.mpsrt_plan = std::move(mpsrt_source.mpsrt_plan);
        result.source = std::move(mpsrt_source.source);
        result.requires_mpsrt_model = mpsrt_source.requires_mpsrt_model;
        return result;
    }

    return make_matmul_msl_fallback_source_plan(module, buffer_manager, node, desc);
}

}  // namespace

KernelSource make_apple_metal_runtime_matmul_kernel_source(mlir::MLIRContext& ctx,
                                                           const GpuBufferManager* buffer_manager,
                                                           const std::shared_ptr<const ov::Node>& node,
                                                           MatMulCodegenDesc desc,
                                                           const ov::Shape& shape_a,
                                                           const ov::Shape& shape_b,
                                                           std::string_view stage_name) {
    auto source_plan = lower_matmul_node_to_metal_kernel_source(ctx,
                                                                buffer_manager,
                                                                node,
                                                                desc,
                                                                shape_a,
                                                                shape_b);
    OPENVINO_ASSERT(source_plan.valid(),
                    "MetalStage: failed to create runtime MatMul source plan for ",
                    stage_name);
    return std::move(source_plan.source);
}

}  // namespace gfx_plugin
}  // namespace ov
