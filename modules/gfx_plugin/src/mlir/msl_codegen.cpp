// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen.hpp"

#include "mlir/gfx_mlir_kernel_builder.hpp"
#include "mlir/gfx_mpsrt_metadata.hpp"
#include "mlir/mlir_kernel_plan_utils.hpp"
#include "transforms/mlir_fused_ops.hpp"

#include "mlir/IR/Builders.h"

#include <sstream>
#include <utility>

namespace ov {
namespace gfx_plugin {
namespace {

bool is_msl_ident_char(char c) {
    return (c >= 'a' && c <= 'z') ||
           (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') ||
           c == '_';
}

bool replace_kernel_entry_name(std::string& source,
                               std::string_view current_entry_point,
                               std::string_view required_entry_point) {
    if (current_entry_point.empty() ||
        required_entry_point.empty() ||
        current_entry_point == required_entry_point) {
        return false;
    }

    const std::string needle = "kernel void " + std::string(current_entry_point);
    size_t pos = source.find(needle);
    while (pos != std::string::npos) {
        const size_t name_pos = pos + std::string("kernel void ").size();
        const size_t after_name = name_pos + current_entry_point.size();
        if (after_name < source.size() && !is_msl_ident_char(source[after_name])) {
            source.replace(name_pos, current_entry_point.size(), required_entry_point);
            return true;
        }
        pos = source.find(needle, pos + 1);
    }
    return false;
}

ov::element::Type resolve_matmul_buffer_type(const ov::element::Type& type,
                                             const ov::element::Type& fallback) {
    if (type != ov::element::dynamic) {
        return type;
    }
    return fallback == ov::element::dynamic ? ov::element::f32 : fallback;
}

void force_apple_msl_buffer_placement(GfxStageOptimizationPlan& plan,
                                      std::string_view stage_type) {
    plan.placement.domain = GfxStageBackendDomain::AppleMsl;
    plan.placement.storage = GfxStageStorageKind::Buffer;
    plan.placement.uses_vendor_primitive = false;
    plan.placement.uses_custom_kernel = true;
    plan.placement.specialization_key = std::string("apple_msl:buffer:") + std::string(stage_type);
}

uint32_t resolve_matmul_source_arg_count(mlir::ModuleOp module,
                                         uint32_t arg_count,
                                         const KernelArgMappingInfo& info) {
    const uint32_t inferred_total =
        static_cast<uint32_t>(infer_kernel_arg_count_from_module(module, info.signature.total()));
    if (arg_count != 0) {
        return arg_count;
    }
    return inferred_total;
}

std::string matmul_epilogue_activation_expr(ActivationKind activation) {
    switch (activation) {
        case ActivationKind::Relu:
            return "max(x, 0.0f)";
        case ActivationKind::Sigmoid:
            return "1.0f / (1.0f + exp(-x))";
        case ActivationKind::Tanh:
            return "tanh(x)";
        case ActivationKind::Gelu:
            return "0.5f * x * (1.0f + tanh(0.79788456f * (x + 0.044715f * x * x * x)))";
        case ActivationKind::Swish:
            return "(x >= 0.0f) ? (x / (1.0f + exp(-x))) : (x * exp(x) / (1.0f + exp(x)))";
        case ActivationKind::HSwish:
            return "x * clamp(x + 3.0f, 0.0f, 6.0f) / 6.0f";
        case ActivationKind::HSigmoid:
            return "clamp(x + 3.0f, 0.0f, 6.0f) / 6.0f";
        case ActivationKind::Abs:
            return "fabs(x)";
        case ActivationKind::Sign:
            return "(x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f)";
        case ActivationKind::Identity:
        default:
            return "x";
    }
}

}  // namespace

std::string normalize_msl_source_for_kernel_plan(std::string source,
                                                 std::string_view current_entry_point,
                                                 const GfxMslKernelPlan& plan) {
    if (!plan.valid || plan.required_entry_point.empty()) {
        return source;
    }
    (void)replace_kernel_entry_name(source, current_entry_point, plan.required_entry_point);
    return source;
}

void configure_msl_kernel_source_for_plan(KernelSource& source,
                                          std::string_view stage_type) {
    if (!source.module) {
        return;
    }

    GfxMpsrtModuleStagePlan stage_plan;
    if (!read_module_mpsrt_stage_plan(source.module, stage_plan) ||
        stage_plan.stage.kind != GfxMpsrtStageKind::MSLDispatch) {
        return;
    }

    auto msl_plan = make_msl_kernel_plan(stage_type, source.entry_point);
    if (!msl_plan.valid) {
        msl_plan = make_msl_kernel_plan(stage_plan.stage.stage_type, source.entry_point);
    }
    if (!msl_plan.valid || msl_plan.required_entry_point.empty()) {
        return;
    }

    const std::string legacy_entry = source.entry_point.empty() ? stage_plan.stage.kernel_name : source.entry_point;
    const std::string required_entry = msl_plan.required_entry_point;
    if (!source.msl_source.empty()) {
        source.msl_source = normalize_msl_source_for_kernel_plan(std::move(source.msl_source),
                                                                 legacy_entry,
                                                                 msl_plan);
    }
    if (source.msl_generator) {
        auto generator = std::move(source.msl_generator);
        source.msl_generator = [generator = std::move(generator), legacy_entry, msl_plan](mlir::ModuleOp module) mutable {
            return normalize_msl_source_for_kernel_plan(generator(module), legacy_entry, msl_plan);
        };
    }
    source.entry_point = required_entry;
    (void)annotate_module_with_mpsrt_external_buffer_abi_from_stage_manifest(source.module,
                                                                            source.signature.arg_count,
                                                                            source.signature.output_arg_count);
}

GfxMpsrtKernelSourcePlan configure_msl_kernel_source_plan(KernelSource source,
                                                          std::string_view stage_type) {
    configure_msl_kernel_source_for_plan(source, stage_type);
    return make_mpsrt_kernel_source_plan_from_configured_source(std::move(source));
}

void configure_msl_kernel_source_for_node(KernelSource& source,
                                          const std::shared_ptr<const ov::Node>& node,
                                          const GpuBufferManager* buffer_manager,
                                          std::string_view stage_type,
                                          bool has_bias,
                                          bool has_activation,
                                          bool has_batchnorm) {
    if (!source.module || !node) {
        return;
    }

    const auto msl_kernel_plan = make_msl_kernel_plan(stage_type, source.entry_point);
    if (!msl_kernel_plan.valid) {
        return;
    }

    auto plan = select_stage_optimization_plan(buffer_manager,
                                               GpuBackend::Metal,
                                               std::string(stage_type),
                                               node,
                                               node->get_output_element_type(0),
                                               has_bias,
                                               has_activation,
                                               has_batchnorm,
                                               GfxStageRuntimeTraits{});
    if (plan.placement.domain != GfxStageBackendDomain::AppleMsl) {
        force_apple_msl_buffer_placement(plan, stage_type);
    }

    annotate_msl_module_with_stage_plan(source.module, plan, std::string(stage_type), source.entry_point);
    auto source_plan = configure_msl_kernel_source_plan(source, stage_type);
    if (source_plan.valid()) {
        source = std::move(source_plan.source);
        return;
    }
    configure_msl_kernel_source_for_plan(source, stage_type);
}

void configure_msl_kernel_source_for_spec(KernelSource& source,
                                          const KernelSpec& spec,
                                          const GpuBufferManager* buffer_manager,
                                          std::string_view entry_point) {
    if (source.entry_point.empty()) {
        source.entry_point = std::string(entry_point);
    }
    configure_msl_kernel_source_for_node(source,
                                         spec.node(),
                                         buffer_manager,
                                         spec.type(),
                                         /*has_bias=*/false,
                                         /*has_activation=*/false,
                                         /*has_batchnorm=*/false);
}

void annotate_msl_module_with_stage_plan(mlir::ModuleOp module,
                                         const GfxStageOptimizationPlan& plan,
                                         const std::string& stage_type,
                                         std::string_view kernel_entry_point) {
    if (!module) {
        return;
    }

    annotate_module_with_mpsrt_stage_plan(module, plan, stage_type, kernel_entry_point);

    GfxMpsrtModuleStagePlan stage_plan;
    if (!read_module_mpsrt_stage_plan(module, stage_plan) ||
        stage_plan.stage.kind != GfxMpsrtStageKind::MSLDispatch) {
        return;
    }

    const auto msl_plan = make_msl_kernel_plan(stage_type, stage_plan.stage.dispatch_entry_point);
    if (!msl_plan.valid) {
        return;
    }

    mlir::Builder builder(module.getContext());
    module->setAttr("gfx.msl.kernel_family", builder.getStringAttr(msl_plan.family_name));
    module->setAttr("gfx.msl.required_entry_point", builder.getStringAttr(msl_plan.required_entry_point));
    module->setAttr("gfx.msl.precompiled_metallib_required",
                    builder.getBoolAttr(msl_plan.precompiled_metallib_required));
    module->setAttr("gfx.msl.threads_per_threadgroup",
                    builder.getI32IntegerAttr(static_cast<int32_t>(msl_plan.threads_per_threadgroup)));
    (void)annotate_module_with_mpsrt_external_buffer_abi_from_stage_manifest(module);
}

std::string generate_msl_for_matmul_mpsrt_epilogue(const MatMulCodegenDesc& desc) {
    const ov::element::Type output_type = resolve_matmul_buffer_type(desc.output_type, desc.element_type);
    const ov::element::Type bias_type = resolve_matmul_buffer_type(desc.bias_type, output_type);
    const std::string scalar_out = msl_type_from_element(output_type);
    const std::string scalar_bias = msl_type_from_element(bias_type);

    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "constant uint BATCH = " << desc.batch << ";\n";
    ss << "constant uint M = " << desc.M << ";\n";
    ss << "constant uint N = " << desc.N << ";\n";
    if (desc.has_bias) {
        ss << "constant uint BIAS_B = " << desc.bias_dims[0] << ";\n";
        ss << "constant uint BIAS_M = " << desc.bias_dims[1] << ";\n";
        ss << "constant uint BIAS_N = " << desc.bias_dims[2] << ";\n";
    }
    ss << "kernel void eltwise_fused_buffer(\n";
    ss << "  device const " << scalar_out << "* gemm [[buffer(0)]],\n";
    if (desc.has_bias) {
        ss << "  device const " << scalar_bias << "* bias [[buffer(1)]],\n";
        ss << "  device " << scalar_out << "* output [[buffer(2)]],\n";
    } else {
        ss << "  device " << scalar_out << "* output [[buffer(1)]],\n";
    }
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "    const uint total = BATCH * M * N;\n";
    ss << "    if (gid >= total) return;\n";
    ss << "    const uint batch = gid / (M * N);\n";
    ss << "    const uint idx = gid - batch * M * N;\n";
    ss << "    const uint row = idx / N;\n";
    ss << "    const uint col = idx - row * N;\n";
    ss << "    float x = static_cast<float>(gemm[gid]);\n";
    if (desc.has_bias) {
        ss << "    const uint bb = (BIAS_B == 1) ? 0 : batch;\n";
        ss << "    const uint bm = (BIAS_M == 1) ? 0 : row;\n";
        ss << "    const uint bn = (BIAS_N == 1) ? 0 : col;\n";
        ss << "    const uint bias_idx = (bb * BIAS_M + bm) * BIAS_N + bn;\n";
        ss << "    x += static_cast<float>(bias[bias_idx]);\n";
    }
    if (desc.has_activation) {
        ss << "    x = " << matmul_epilogue_activation_expr(desc.activation) << ";\n";
    }
    ss << "    output[gid] = static_cast<" << scalar_out << ">(x);\n";
    ss << "}\n";
    return ss.str();
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
    annotate_msl_module_with_stage_plan(module, plan, kStageType);

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

    auto source_desc = desc;
    auto source = plan_ctx.build_info.plan.to_source_with_msl_generator(
        [source_desc](mlir::ModuleOp mod) {
            return generate_msl_from_mlir(mod, source_desc);
        });
    source.signature.output_arg_count = 1;
    configure_msl_kernel_source_for_plan(source, kStageType);

    result.kind = GfxMatMulMetalKernelSourcePlanKind::MslFallback;
    result.source = std::move(source);
    result.requires_mpsrt_model = false;
    auto mpsrt_plan = make_mpsrt_kernel_source_plan_from_configured_source(result.source);
    if (mpsrt_plan.valid()) {
        result.mpsrt_plan = std::move(mpsrt_plan);
        result.source = result.mpsrt_plan.source;
        result.requires_mpsrt_model = result.mpsrt_plan.requires_mpsrt_model;
    }
    return result;
}

GfxMatMulMpsrtKernelSourcePlan lower_matmul_module_to_mpsrt_kernel_source(
    mlir::ModuleOp module,
    const GfxStageOptimizationPlan& plan,
    const MatMulCodegenDesc& desc,
    const ov::Shape& shape_a,
    const ov::Shape& shape_b) {
    GfxMatMulMpsrtKernelSourcePlan result{};
    result.lowering = annotate_module_with_matmul_mpsrt_plan(module, plan, desc, shape_a, shape_b);
    if (result.lowering == GfxMatMulMpsrtLoweringKind::None) {
        return result;
    }

    GfxMpsrtKernelSourceOptions source_options{};
    switch (result.lowering) {
        case GfxMatMulMpsrtLoweringKind::MpsGemm:
            break;
        case GfxMatMulMpsrtLoweringKind::MpsGemmWithMslEpilogue:
            source_options.msl_source = generate_msl_for_matmul_mpsrt_epilogue(desc);
            break;
        case GfxMatMulMpsrtLoweringKind::None:
            break;
    }
    result.mpsrt_plan = make_mpsrt_kernel_source_plan_from_module(module, std::move(source_options));
    if (!result.mpsrt_plan.valid()) {
        result.lowering = GfxMatMulMpsrtLoweringKind::None;
        return result;
    }
    result.source = result.mpsrt_plan.source;
    result.requires_mpsrt_model = result.mpsrt_plan.requires_mpsrt_model;
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

}  // namespace gfx_plugin
}  // namespace ov
