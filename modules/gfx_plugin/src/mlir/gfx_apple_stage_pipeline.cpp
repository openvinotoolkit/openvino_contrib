// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/gfx_apple_stage_pipeline.hpp"

#include <algorithm>
#include <utility>

#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/gfx_mpsrt_runtime_abi_pipeline.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

std::vector<GfxAppleStagePipelinePassKind> make_apple_stage_pipeline_boundaries(
    bool materialize_typed_program) {
    std::vector<GfxAppleStagePipelinePassKind> passes{
        GfxAppleStagePipelinePassKind::CoreCanonicalize,
        GfxAppleStagePipelinePassKind::Placement,
        GfxAppleStagePipelinePassKind::StorageAssignment,
        GfxAppleStagePipelinePassKind::Fusion,
        GfxAppleStagePipelinePassKind::VendorDescriptor,
        GfxAppleStagePipelinePassKind::StageManifest,
    };
    if (materialize_typed_program) {
        passes.push_back(GfxAppleStagePipelinePassKind::RuntimeAbi);
        passes.push_back(GfxAppleStagePipelinePassKind::RuntimeAbiCallPlan);
    }
    return passes;
}

void annotate_apple_stage_pipeline_boundaries(
    mlir::ModuleOp module,
    const std::vector<GfxAppleStagePipelinePassKind>& passes) {
    if (!module) {
        return;
    }
    mlir::Builder builder(module.getContext());
    module->setAttr("gfx.apple.pipeline.pass_boundary_count",
                    builder.getI32IntegerAttr(static_cast<int32_t>(passes.size())));
    for (size_t i = 0; i < passes.size(); ++i) {
        const std::string prefix =
            "gfx.apple.pipeline.pass" + std::to_string(i);
        module->setAttr(prefix + ".name",
                        builder.getStringAttr(
                            gfx_apple_stage_pipeline_pass_name(passes[i])));
    }
}

void annotate_apple_program_pipeline(
    mlir::ModuleOp module,
    const GfxMpsrtProgram& program) {
    if (!module) {
        return;
    }
    annotate_apple_stage_pipeline_boundaries(
        module,
        gfx_apple_stage_pipeline_pass_boundaries(/*materialize_typed_program=*/true));

    mlir::Builder builder(module.getContext());
    module->setAttr("gfx.apple.pipeline.program.kind",
                    builder.getStringAttr(program.multi_stage ? "multi_stage" : "single_stage"));
    module->setAttr("gfx.apple.pipeline.program.record_key",
                    builder.getStringAttr(program.record_key));
    module->setAttr("gfx.apple.pipeline.program.stage_count",
                    builder.getI32IntegerAttr(static_cast<int32_t>(program.stages.size())));
    for (size_t i = 0; i < program.stages.size(); ++i) {
        const auto& stage = program.stages[i].stage;
        const std::string prefix = "gfx.apple.pipeline.program.stage" + std::to_string(i);
        module->setAttr(prefix + ".kind",
                        builder.getStringAttr(gfx_mpsrt_stage_kind_name(stage.kind)));
        if (stage.stage_manifest.valid) {
            module->setAttr(prefix + ".backend_domain",
                            builder.getStringAttr(
                                gfx_kernel_backend_domain_name(stage.stage_manifest.backend_domain)));
            module->setAttr(prefix + ".execution_kind",
                            builder.getStringAttr(
                                gfx_kernel_execution_kind_name(stage.stage_manifest.execution_kind)));
            module->setAttr(prefix + ".storage",
                            builder.getStringAttr(
                                gfx_kernel_storage_kind_name(stage.stage_manifest.storage)));
        } else {
            module->setAttr(prefix + ".backend_domain",
                            builder.getStringAttr(gfx_stage_backend_domain_name(stage.domain)));
            module->setAttr(prefix + ".execution_kind",
                            builder.getStringAttr(stage.uses_vendor_primitive
                                                      ? "vendor_primitive"
                                                      : "custom_kernel"));
            module->setAttr(prefix + ".storage",
                            builder.getStringAttr(gfx_mpsrt_storage_name(stage.output_storage)));
        }
    }
}

bool validate_apple_program_stage_contract(const GfxMpsrtBuilderStageSpec& spec) {
    const auto& stage = spec.stage;
    if (stage.stage_manifest.valid) {
        if (stage.domain == GfxStageBackendDomain::AppleMps &&
            (stage.stage_manifest.backend_domain != GfxKernelBackendDomain::AppleMps ||
             stage.stage_manifest.execution_kind != GfxKernelExecutionKind::VendorPrimitive)) {
            return false;
        }
        if (stage.domain == GfxStageBackendDomain::AppleMsl &&
            (stage.stage_manifest.backend_domain != GfxKernelBackendDomain::AppleMsl ||
             stage.stage_manifest.execution_kind != GfxKernelExecutionKind::CustomKernel)) {
            return false;
        }
    }
    return stage.domain == GfxStageBackendDomain::AppleMps ||
           stage.domain == GfxStageBackendDomain::AppleMsl;
}

bool finalize_apple_program_stage_desc(GfxMpsrtStageDesc& stage) {
    if (stage.kind == GfxMpsrtStageKind::Unknown && stage.stage_manifest.valid) {
        stage.kind = gfx_mpsrt_stage_kind_from_manifest(stage.stage_manifest);
    }
    if (stage.domain == GfxStageBackendDomain::Unknown && stage.stage_manifest.valid) {
        switch (stage.stage_manifest.backend_domain) {
            case GfxKernelBackendDomain::AppleMps:
                stage.domain = GfxStageBackendDomain::AppleMps;
                break;
            case GfxKernelBackendDomain::AppleMsl:
                stage.domain = GfxStageBackendDomain::AppleMsl;
                break;
            case GfxKernelBackendDomain::Spirv:
                stage.domain = GfxStageBackendDomain::Spirv;
                break;
            case GfxKernelBackendDomain::Unknown:
            default:
                break;
        }
    }
    if (stage.stage_type.empty() && stage.stage_manifest.valid) {
        stage.stage_type = gfx_mpsrt_stage_type_from_manifest(stage.stage_manifest);
    }
    if (stage.builder_symbol.empty()) {
        stage.builder_symbol = gfx_mpsrt_builder_symbol(stage.kind);
    }
    if (stage.kind == GfxMpsrtStageKind::MSLDispatch) {
        const auto dispatch =
            gfx_mpsrt_custom_dispatch_spec_from_kernel_manifest(stage.stage_manifest.custom_kernel);
        if (dispatch.valid) {
            stage.dispatch_kernel_family = dispatch.kernel_family;
            stage.dispatch_entry_point = dispatch.entry_point;
            stage.dispatch_kernel_family_id = dispatch.kernel_family_id;
            stage.dispatch_flags = dispatch.flags;
            stage.dispatch_threads_per_threadgroup = dispatch.threads_per_threadgroup;
            stage.dispatch_precompiled_kernel_required = dispatch.precompiled_binary_required;
            stage.kernel_name = dispatch.entry_point;
        }
    }
    if (stage.kernel_name.empty()) {
        stage.kernel_name = gfx_mpsrt_default_kernel_name(stage.kind, stage.stage_type);
    }
    if (stage.specialization_key.empty() && stage.stage_manifest.valid) {
        stage.specialization_key = stage.stage_manifest.specialization_key;
    }
    return stage.kind != GfxMpsrtStageKind::Unknown &&
           stage.domain != GfxStageBackendDomain::Unknown &&
           !stage.builder_symbol.empty();
}

GfxMpsrtProgram make_apple_mpsrt_program_from_plan(
    const GfxAppleMpsrtProgramPlan& plan) {
    GfxMpsrtProgram program{};
    if (plan.record_key.empty() || plan.stages.empty()) {
        return program;
    }

    program.record_key = plan.record_key;
    program.multi_stage = plan.stages.size() > 1;
    program.inputs = plan.inputs;
    program.output_values = plan.output_values;
    program.external_buffer_abi = plan.external_buffer_abi;
    program.has_storage_bridges = plan.has_storage_bridges;
    program.storage_bridges = plan.storage_bridges;

    for (const auto& request : plan.stages) {
        if (request.inputs.empty() ||
            request.outputs.empty() ||
            request.outputs.size() != request.output_descs.size()) {
            program = {};
            return program;
        }
        auto stage = request.stage;
        if (!finalize_apple_program_stage_desc(stage)) {
            program = {};
            return program;
        }
        const auto stage_record_key = gfx_mpsrt_stage_record_key(stage);
        if (stage_record_key.empty()) {
            program = {};
            return program;
        }
        program.stages.push_back({std::move(stage),
                                  stage_record_key,
                                  request.inputs,
                                  request.outputs,
                                  request.output_descs});
    }

    if (program.output_values.empty()) {
        program.output_values = program.stages.back().outputs;
    }

    if (!program.external_buffer_abi.valid) {
        program.external_buffer_abi =
            gfx_mpsrt_make_external_io_abi(program.inputs.size(), program.output_values.size());
    } else {
        auto abi = program.external_buffer_abi;
        if (!gfx_mpsrt_finalize_external_buffer_abi(abi)) {
            program = {};
            return program;
        }
        program.external_buffer_abi = std::move(abi);
    }

    program.valid = gfx_mpsrt_validate_program(program, nullptr);
    if (!program.valid) {
        program = {};
        return program;
    }

    if (!program.has_storage_bridges) {
        GfxMpsrtBuilderPlan builder_plan{};
        if (gfx_mpsrt_build_builder_plan_from_program(program, builder_plan) &&
            !builder_plan.storage_bridges.empty()) {
            program.has_storage_bridges = true;
            program.storage_bridges = std::move(builder_plan.storage_bridges);
        }
    }
    return program;
}

GfxMpsrtStageDesc make_apple_mps_gemm_stage_desc(const GfxMpsrtGemmAbiDesc& gemm) {
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
    stage.gemm_desc = gemm;
    stage.gemm_desc.alpha = stage.gemm_desc.alpha == 0.0f ? 1.0f : stage.gemm_desc.alpha;
    stage.stage_manifest = make_gfx_vendor_stage_manifest(GfxKernelStageFamily::Gemm,
                                                          GfxKernelBackendDomain::AppleMps,
                                                          GfxKernelStorageKind::Matrix,
                                                          stage.specialization_key);
    return stage;
}

GfxMpsrtStageDesc make_apple_msl_gemm_epilogue_stage_desc(bool has_bias) {
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
                                        static_cast<uint32_t>(GfxKernelFamily::EltwiseFusedBuffer),
                                        "eltwise_fused_buffer",
                                        has_bias
                                            ? make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorInput,
                                                                         GfxKernelBufferRole::TensorInput,
                                                                         GfxKernelBufferRole::TensorOutput})
                                            : make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorInput,
                                                                         GfxKernelBufferRole::TensorOutput}),
                                        make_gfx_kernel_linear_dispatch_policy(
                                            256,
                                            /*precompiled_binary_required=*/true)));
    (void)finalize_apple_program_stage_desc(stage);
    return stage;
}

const char* vendor_descriptor_kind_name(
    GfxAppleStagePipelineOptions::VendorPrimitiveDescriptor::Kind kind) {
    using Kind = GfxAppleStagePipelineOptions::VendorPrimitiveDescriptor::Kind;
    switch (kind) {
        case Kind::None:
            return "none";
        case Kind::Gemm:
            return "gemm";
        case Kind::Conv2D:
            return "conv2d";
        case Kind::Pool2D:
            return "pool2d";
        case Kind::Resize2D:
            return "resize2d";
        case Kind::Softmax:
            return "softmax";
        case Kind::TopK:
            return "topk";
    }
    return "unknown";
}

bool apple_vendor_descriptor_matches_stage(
    GfxMpsrtStageKind stage_kind,
    GfxAppleStagePipelineOptions::VendorPrimitiveDescriptor::Kind descriptor_kind) {
    using Kind = GfxAppleStagePipelineOptions::VendorPrimitiveDescriptor::Kind;
    switch (descriptor_kind) {
        case Kind::None:
            return true;
        case Kind::Gemm:
            return stage_kind == GfxMpsrtStageKind::MPSGemm;
        case Kind::Conv2D:
            return stage_kind == GfxMpsrtStageKind::MPSConv2D ||
                   stage_kind == GfxMpsrtStageKind::MPSGroupConv2D;
        case Kind::Pool2D:
            return stage_kind == GfxMpsrtStageKind::MPSPool2D;
        case Kind::Resize2D:
            return stage_kind == GfxMpsrtStageKind::MPSResize2D;
        case Kind::Softmax:
            return stage_kind == GfxMpsrtStageKind::MPSSoftmax;
        case Kind::TopK:
            return stage_kind == GfxMpsrtStageKind::MPSTopK;
    }
    return false;
}

bool apply_apple_vendor_descriptor(
    GfxMpsrtModuleStagePlan& stage_plan,
    const GfxAppleStagePipelineOptions::VendorPrimitiveDescriptor& descriptor) {
    using Kind = GfxAppleStagePipelineOptions::VendorPrimitiveDescriptor::Kind;
    if (descriptor.kind == Kind::None) {
        return true;
    }
    if (!stage_plan.valid ||
        !gfx_mpsrt_stage_is_apple_mps_vendor(stage_plan.stage) ||
        !apple_vendor_descriptor_matches_stage(stage_plan.stage.kind, descriptor.kind)) {
        return false;
    }

    switch (descriptor.kind) {
        case Kind::Gemm:
            stage_plan.stage.gemm_desc = descriptor.gemm;
            break;
        case Kind::Conv2D:
            if (stage_plan.inputs.size() <= 1 ||
                (stage_plan.inputs[1].flags & GfxMpsrtTensorFlagConst) == 0 ||
                stage_plan.inputs[1].storage != GfxMpsrtStorage::Buffer) {
                return false;
            }
            stage_plan.stage.conv2d_desc = descriptor.conv2d;
            break;
        case Kind::Pool2D:
            stage_plan.stage.pool2d_desc = descriptor.pool2d;
            break;
        case Kind::Resize2D:
            stage_plan.stage.resize2d_desc = descriptor.resize2d;
            break;
        case Kind::Softmax:
            stage_plan.stage.softmax_desc = descriptor.softmax;
            break;
        case Kind::TopK:
            stage_plan.stage.topk_desc = descriptor.topk;
            break;
        case Kind::None:
            break;
    }

    stage_plan.stage_record_key = gfx_mpsrt_stage_record_key(stage_plan.stage);
    stage_plan.valid = !stage_plan.stage_record_key.empty();
    return stage_plan.valid;
}

bool apply_apple_stage_tensor_desc_overrides(GfxMpsrtModuleStagePlan& stage_plan,
                                             const GfxAppleStagePipelineOptions& options) {
    if (!stage_plan.valid) {
        return false;
    }
    if (options.input_descs.empty() && options.output_descs.empty()) {
        return true;
    }
    if (!options.input_descs.empty()) {
        if (stage_plan.inputs.size() != options.input_descs.size()) {
            return false;
        }
        stage_plan.inputs = options.input_descs;
    }
    if (!options.output_descs.empty()) {
        if (stage_plan.outputs.size() != options.output_descs.size()) {
            return false;
        }
        stage_plan.outputs = options.output_descs;
    }

    stage_plan.stage.input_storage = !stage_plan.inputs.empty()
                                         ? stage_plan.inputs.front().storage
                                         : GfxMpsrtStorage::Unknown;
    stage_plan.stage.output_storage = !stage_plan.outputs.empty()
                                          ? stage_plan.outputs.front().storage
                                          : stage_plan.stage.input_storage;
    stage_plan.stage.layout = stage_plan.stage.output_storage != GfxMpsrtStorage::Unknown
                                  ? gfx_mpsrt_stage_layout_for_storage(stage_plan.stage.output_storage)
                                  : GfxMpsrtLayout::Unknown;
    stage_plan.stage_record_key = gfx_mpsrt_stage_record_key(stage_plan.stage);
    stage_plan.valid = !stage_plan.stage_record_key.empty();
    return stage_plan.valid;
}

GfxAppleProgramPipelineResult materialize_apple_mps_vendor_program(
    mlir::ModuleOp module,
    const GfxStageOptimizationPlan& plan,
    const std::string& stage_type,
    const GfxAppleStagePipelineOptions::VendorPrimitiveDescriptor& vendor_descriptor,
    const std::vector<GfxKernelBufferRole>& semantic_input_roles,
    const GfxMpsrtExternalBufferAbiPlan& external_buffer_abi,
    const std::vector<GfxMpsrtTensorDesc>& input_descs,
    const std::vector<GfxMpsrtTensorDesc>& output_descs) {
    GfxAppleProgramPipelineResult result{};
    if (!module || !external_buffer_abi.valid) {
        return result;
    }

    GfxAppleStagePipelineOptions options{};
    options.plan = plan;
    options.stage_type = stage_type;
    options.semantic_input_roles = semantic_input_roles;
    options.input_descs = input_descs;
    options.output_descs = output_descs;
    options.vendor_descriptor = vendor_descriptor;
    options.materialize_typed_program = false;
    const auto stage_pipeline = run_gfx_apple_stage_pipeline(module, options);
    if (!stage_pipeline.valid ||
        !gfx_mpsrt_stage_is_apple_mps_vendor(stage_pipeline.stage_plan.stage)) {
        return result;
    }

    const auto input_values =
        gfx_mpsrt_make_sequential_values(stage_pipeline.stage_plan.inputs.size());
    const auto output_values =
        gfx_mpsrt_make_sequential_values(stage_pipeline.stage_plan.outputs.size(),
                                         static_cast<GfxMpsrtValue>(input_values.size()));
    GfxAppleMpsrtProgramPlan program_plan{};
    program_plan.record_key = stage_pipeline.stage_plan.stage_record_key;
    program_plan.inputs = stage_pipeline.stage_plan.inputs;
    program_plan.output_values = output_values;
    program_plan.external_buffer_abi = external_buffer_abi;
    program_plan.stages.push_back({stage_pipeline.stage_plan.stage,
                                   input_values,
                                   output_values,
                                   stage_pipeline.stage_plan.outputs});
    if (read_module_apple_storage_assignment(module, program_plan.storage_bridges)) {
        program_plan.has_storage_bridges = true;
    }

    return materialize_apple_mpsrt_program_plan(module, program_plan);
}

const char* execution_kind_name(const GfxStagePlacementPlan& placement) {
    return placement.uses_vendor_primitive ? "vendor_primitive" : "custom_kernel";
}

const char* tensor_layout_name(GfxTensorLayoutKind kind) {
    switch (kind) {
        case GfxTensorLayoutKind::Materialized:
            return "materialized";
        case GfxTensorLayoutKind::ViewOnly:
            return "view_only";
        case GfxTensorLayoutKind::Unknown:
        default:
            return "unknown";
    }
}

class GfxApplePlacementPass final
    : public mlir::PassWrapper<GfxApplePlacementPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GfxApplePlacementPass)

    explicit GfxApplePlacementPass(GfxAppleStagePipelineOptions options)
        : m_options(std::move(options)) {}

    llvm::StringRef getArgument() const final {
        return "gfx-apple-placement";
    }

    llvm::StringRef getDescription() const final {
        return "Materialize Apple backend placement metadata before stage manifest lowering";
    }

    void runOnOperation() final {
        auto module = getOperation();
        auto* ctx = module.getContext();
        const auto& placement = m_options.plan.placement;
        module->setAttr("gfx.apple.pipeline.stage_type",
                        mlir::StringAttr::get(ctx, m_options.stage_type));
        module->setAttr("gfx.apple.pipeline.placement.backend_domain",
                        mlir::StringAttr::get(ctx, gfx_stage_backend_domain_name(placement.domain)));
        module->setAttr("gfx.apple.pipeline.placement.execution_kind",
                        mlir::StringAttr::get(ctx, execution_kind_name(placement)));
        module->setAttr("gfx.apple.pipeline.placement.storage",
                        mlir::StringAttr::get(ctx, gfx_stage_storage_kind_name(placement.storage)));
        module->setAttr("gfx.apple.pipeline.placement.specialization_key",
                        mlir::StringAttr::get(ctx, placement.specialization_key));
    }

private:
    GfxAppleStagePipelineOptions m_options;
};

class GfxAppleStorageAssignmentPass final
    : public mlir::PassWrapper<GfxAppleStorageAssignmentPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GfxAppleStorageAssignmentPass)

    explicit GfxAppleStorageAssignmentPass(GfxAppleStagePipelineOptions options)
        : m_options(std::move(options)) {}

    llvm::StringRef getArgument() const final {
        return "gfx-apple-storage-assignment";
    }

    llvm::StringRef getDescription() const final {
        return "Materialize Apple storage contract metadata before stage manifest lowering";
    }

    void runOnOperation() final {
        auto module = getOperation();
        auto* ctx = module.getContext();
        module->setAttr("gfx.apple.pipeline.storage.contract",
                        mlir::StringAttr::get(ctx, gfx_stage_storage_kind_name(m_options.plan.placement.storage)));
        module->setAttr("gfx.apple.pipeline.storage.view_only",
                        mlir::BoolAttr::get(ctx, m_options.plan.layout.view_only));
        module->setAttr("gfx.apple.pipeline.storage.layout",
                        mlir::StringAttr::get(ctx, tensor_layout_name(m_options.plan.layout.kind)));

        GfxMpsrtModuleStagePlan stage_plan{};
        if (!build_module_mpsrt_stage_plan(module,
                                           m_options.plan,
                                           m_options.stage_type,
                                           m_options.kernel_entry_point,
                                           stage_plan,
                                           m_options.semantic_input_roles) ||
            !apply_apple_stage_tensor_desc_overrides(stage_plan, m_options)) {
            module->emitError("failed to assign Apple storage bridges for ")
                << m_options.stage_type;
            signalPassFailure();
            return;
        }
        if (stage_plan.stage.input_storage == GfxMpsrtStorage::Buffer &&
            stage_plan.stage.output_storage == GfxMpsrtStorage::Buffer) {
            annotate_module_with_empty_apple_storage_assignment(module);
            return;
        }
        if (!annotate_module_with_apple_storage_assignment(module, stage_plan)) {
            module->emitError("failed to assign Apple storage bridges for ")
                << m_options.stage_type;
            signalPassFailure();
        }
    }

private:
    GfxAppleStagePipelineOptions m_options;
};

class GfxAppleFusionPass final
    : public mlir::PassWrapper<GfxAppleFusionPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GfxAppleFusionPass)

    explicit GfxAppleFusionPass(GfxAppleStagePipelineOptions options)
        : m_options(std::move(options)) {}

    llvm::StringRef getArgument() const final {
        return "gfx-apple-fusion";
    }

    llvm::StringRef getDescription() const final {
        return "Materialize Apple post-op fusion contract metadata before stage manifest lowering";
    }

    void runOnOperation() final {
        auto module = getOperation();
        auto* ctx = module.getContext();
        module->setAttr("gfx.apple.pipeline.fusion.bias",
                        mlir::BoolAttr::get(ctx, m_options.plan.post_ops.bias));
        module->setAttr("gfx.apple.pipeline.fusion.activation",
                        mlir::BoolAttr::get(ctx, m_options.plan.post_ops.activation));
        module->setAttr("gfx.apple.pipeline.fusion.batchnorm",
                        mlir::BoolAttr::get(ctx, m_options.plan.post_ops.batchnorm));
    }

private:
    GfxAppleStagePipelineOptions m_options;
};

class GfxAppleVendorDescriptorPass final
    : public mlir::PassWrapper<GfxAppleVendorDescriptorPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GfxAppleVendorDescriptorPass)

    explicit GfxAppleVendorDescriptorPass(GfxAppleStagePipelineOptions options)
        : m_options(std::move(options)) {}

    llvm::StringRef getArgument() const final {
        return "gfx-apple-vendor-descriptor";
    }

    llvm::StringRef getDescription() const final {
        return "Validate Apple MPS vendor primitive descriptors before stage manifest lowering";
    }

    void runOnOperation() final {
        auto module = getOperation();
        auto* ctx = module.getContext();
        module->setAttr("gfx.apple.pipeline.vendor_descriptor.kind",
                        mlir::StringAttr::get(
                            ctx,
                            vendor_descriptor_kind_name(
                                m_options.vendor_descriptor.kind)));
        if (m_options.vendor_descriptor.kind ==
            GfxAppleStagePipelineOptions::VendorPrimitiveDescriptor::Kind::None) {
            return;
        }

        GfxMpsrtModuleStagePlan stage_plan{};
        if (!build_module_mpsrt_stage_plan(module,
                                           m_options.plan,
                                           m_options.stage_type,
                                           m_options.kernel_entry_point,
                                           stage_plan,
                                           m_options.semantic_input_roles) ||
            !apply_apple_stage_tensor_desc_overrides(stage_plan, m_options) ||
            !apply_apple_vendor_descriptor(stage_plan,
                                           m_options.vendor_descriptor)) {
            module->emitError("failed to enrich Apple vendor descriptor for ")
                << m_options.stage_type;
            signalPassFailure();
        }
    }

private:
    GfxAppleStagePipelineOptions m_options;
};

class GfxAppleStageManifestPass final
    : public mlir::PassWrapper<GfxAppleStageManifestPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GfxAppleStageManifestPass)

    explicit GfxAppleStageManifestPass(GfxAppleStagePipelineOptions options)
        : m_options(std::move(options)) {}

    llvm::StringRef getArgument() const final {
        return "gfx-apple-stage-manifest";
    }

    llvm::StringRef getDescription() const final {
        return "Lower Apple placement, storage, and fusion metadata to canonical gfx.stage_manifest";
    }

    void runOnOperation() final {
        GfxMpsrtModuleStagePlan stage_plan{};
        auto module = getOperation();
        if (!build_module_mpsrt_stage_plan(module,
                                           m_options.plan,
                                           m_options.stage_type,
                                           m_options.kernel_entry_point,
                                           stage_plan,
                                           m_options.semantic_input_roles) ||
            !apply_apple_stage_tensor_desc_overrides(stage_plan, m_options) ||
            !apply_apple_vendor_descriptor(stage_plan,
                                           m_options.vendor_descriptor)) {
            getOperation()->emitError("failed to materialize Apple stage manifest for ")
                << m_options.stage_type;
            signalPassFailure();
            return;
        }
        detail::gfx_mpsrt_set_stage_manifest_attrs(module,
                                                   stage_plan.stage.stage_manifest);
    }

private:
    GfxAppleStagePipelineOptions m_options;
};

class GfxAppleRuntimeAbiPass final
    : public mlir::PassWrapper<GfxAppleRuntimeAbiPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GfxAppleRuntimeAbiPass)

    explicit GfxAppleRuntimeAbiPass(GfxAppleStagePipelineOptions options)
        : m_options(std::move(options)) {}

    llvm::StringRef getArgument() const final {
        return "gfx-apple-runtime-abi";
    }

    llvm::StringRef getDescription() const final {
        return "Materialize canonical Apple stage manifests to the typed MPSRT runtime ABI facade";
    }

    void runOnOperation() final {
        auto module = getOperation();
        auto* ctx = module.getContext();
        GfxMpsrtModuleStagePlan stage_plan{};
        if (!build_module_mpsrt_stage_plan(module,
                                           m_options.plan,
                                           m_options.stage_type,
                                           m_options.kernel_entry_point,
                                           stage_plan,
                                           m_options.semantic_input_roles) ||
            !apply_apple_stage_tensor_desc_overrides(stage_plan, m_options) ||
            !apply_apple_vendor_descriptor(stage_plan,
                                           m_options.vendor_descriptor)) {
            signalPassFailure();
            return;
        }

        GfxMpsrtExternalBufferAbiPlan external_buffer_abi{};
        if (stage_plan.stage.uses_custom_kernel) {
            (void)gfx_mpsrt_external_buffer_abi_from_kernel_manifest(module,
                                                                    external_buffer_abi);
        }
        if (!materialize_module_mpsrt_ops_from_stage_plan(module,
                                                          stage_plan,
                                                          external_buffer_abi)) {
            signalPassFailure();
            return;
        }
        module->setAttr("gfx.apple.pipeline.runtime_abi.typed_program_materialized",
                        mlir::BoolAttr::get(ctx, true));
    }

private:
    GfxAppleStagePipelineOptions m_options;
};

}  // namespace

const char* gfx_apple_stage_pipeline_pass_name(GfxAppleStagePipelinePassKind kind) {
    switch (kind) {
        case GfxAppleStagePipelinePassKind::CoreCanonicalize:
            return "gfx-core-canonicalize";
        case GfxAppleStagePipelinePassKind::Placement:
            return "gfx-apple-placement";
        case GfxAppleStagePipelinePassKind::StorageAssignment:
            return "gfx-apple-storage-assignment";
        case GfxAppleStagePipelinePassKind::Fusion:
            return "gfx-apple-fusion";
        case GfxAppleStagePipelinePassKind::VendorDescriptor:
            return "gfx-apple-vendor-descriptor";
        case GfxAppleStagePipelinePassKind::StageManifest:
            return "gfx-apple-stage-manifest";
        case GfxAppleStagePipelinePassKind::RuntimeAbi:
            return "gfx-apple-runtime-abi";
        case GfxAppleStagePipelinePassKind::RuntimeAbiCallPlan:
            return "gfx-apple-runtime-abi-call-plan";
    }
    return "gfx-apple-unknown";
}

std::vector<GfxAppleStagePipelinePassKind> gfx_apple_stage_pipeline_pass_boundaries(
    bool materialize_typed_program) {
    return make_apple_stage_pipeline_boundaries(materialize_typed_program);
}

std::unique_ptr<mlir::Pass> createGfxAppleCanonicalizePass() {
    return mlir::createCanonicalizerPass();
}

std::unique_ptr<mlir::Pass> createGfxApplePlacementPass(
    const GfxAppleStagePipelineOptions& options) {
    return std::make_unique<GfxApplePlacementPass>(options);
}

std::unique_ptr<mlir::Pass> createGfxAppleStorageAssignmentPass(
    const GfxAppleStagePipelineOptions& options) {
    return std::make_unique<GfxAppleStorageAssignmentPass>(options);
}

std::unique_ptr<mlir::Pass> createGfxAppleFusionPass(
    const GfxAppleStagePipelineOptions& options) {
    return std::make_unique<GfxAppleFusionPass>(options);
}

std::unique_ptr<mlir::Pass> createGfxAppleVendorDescriptorPass(
    const GfxAppleStagePipelineOptions& options) {
    return std::make_unique<GfxAppleVendorDescriptorPass>(options);
}

std::unique_ptr<mlir::Pass> createGfxAppleStageManifestPass(
    const GfxAppleStagePipelineOptions& options) {
    return std::make_unique<GfxAppleStageManifestPass>(options);
}

std::unique_ptr<mlir::Pass> createGfxAppleRuntimeAbiPass(
    const GfxAppleStagePipelineOptions& options) {
    return std::make_unique<GfxAppleRuntimeAbiPass>(options);
}

GfxAppleStagePipelineResult run_gfx_apple_stage_pipeline(
    mlir::ModuleOp module,
    const GfxAppleStagePipelineOptions& options) {
    GfxAppleStagePipelineResult result{};
    if (!module || options.stage_type.empty()) {
        return result;
    }

    annotate_apple_stage_pipeline_boundaries(
        module,
        gfx_apple_stage_pipeline_pass_boundaries(options.materialize_typed_program));

    mlir::PassManager pm(module.getContext());
    pm.addPass(createGfxAppleCanonicalizePass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(createGfxApplePlacementPass(options));
    pm.addPass(createGfxAppleStorageAssignmentPass(options));
    pm.addPass(createGfxAppleFusionPass(options));
    pm.addPass(createGfxAppleVendorDescriptorPass(options));
    pm.addPass(createGfxAppleStageManifestPass(options));
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    if (options.materialize_typed_program) {
        pm.addPass(createGfxAppleRuntimeAbiPass(options));
        pm.addPass(createGfxAppleMpsrtRuntimeAbiCallPlanPass());
    }
    if (mlir::failed(pm.run(module))) {
        return result;
    }

    if (!build_module_mpsrt_stage_plan(module,
                                       options.plan,
                                       options.stage_type,
                                       options.kernel_entry_point,
                                       result.stage_plan,
                                       options.semantic_input_roles) ||
        !apply_apple_stage_tensor_desc_overrides(result.stage_plan, options)) {
        return result;
    }
    if (!apply_apple_vendor_descriptor(result.stage_plan,
                                       options.vendor_descriptor)) {
        result = {};
        return result;
    }
    result.valid = true;

    if (options.materialize_typed_program) {
        GfxMpsrtProgram typed_program{};
        result.typed_program_materialized = read_module_mpsrt_program(module, typed_program);
        if (!result.typed_program_materialized) {
            result = {};
            return result;
        }
        if (!has_gfx_apple_mpsrt_runtime_abi_call_plan(module)) {
            result = {};
            return result;
        }
    }
    return result;
}

GfxAppleProgramPipelineResult materialize_apple_mps_conv2d_program(
    mlir::ModuleOp module,
    const GfxStageOptimizationPlan& plan,
    const std::string& stage_type,
    const GfxMpsrtConv2DAbiDesc& desc,
    const std::vector<GfxKernelBufferRole>& semantic_input_roles) {
    GfxAppleStagePipelineOptions::VendorPrimitiveDescriptor vendor_descriptor{};
    vendor_descriptor.kind =
        GfxAppleStagePipelineOptions::VendorPrimitiveDescriptor::Kind::Conv2D;
    vendor_descriptor.conv2d = desc;
    const auto roles =
        semantic_input_roles.empty()
            ? std::vector<GfxKernelBufferRole>{GfxKernelBufferRole::TensorInput,
                                               GfxKernelBufferRole::ConstTensor}
            : semantic_input_roles;
    return materialize_apple_mps_vendor_program(
        module,
        plan,
        stage_type,
        vendor_descriptor,
        roles,
        gfx_mpsrt_make_external_buffer_abi_from_roles({GfxMpsrtExternalBufferRole::TensorInput,
                                                       GfxMpsrtExternalBufferRole::ConstBuffer,
                                                       GfxMpsrtExternalBufferRole::ConstBuffer,
                                                       GfxMpsrtExternalBufferRole::ConstBuffer,
                                                       GfxMpsrtExternalBufferRole::ConstBuffer,
                                                       GfxMpsrtExternalBufferRole::ConstBuffer,
                                                       GfxMpsrtExternalBufferRole::ConstBuffer,
                                                       GfxMpsrtExternalBufferRole::RuntimeParams,
                                                       GfxMpsrtExternalBufferRole::TensorOutput}),
        {},
        {});
}

GfxAppleProgramPipelineResult materialize_apple_mps_pool2d_program(
    mlir::ModuleOp module,
    const GfxStageOptimizationPlan& plan,
    const std::string& stage_type,
    const GfxMpsrtPool2DAbiDesc& desc,
    const std::vector<GfxKernelBufferRole>& semantic_input_roles,
    const std::vector<GfxMpsrtTensorDesc>& input_descs,
    const std::vector<GfxMpsrtTensorDesc>& output_descs) {
    GfxAppleStagePipelineOptions::VendorPrimitiveDescriptor vendor_descriptor{};
    vendor_descriptor.kind =
        GfxAppleStagePipelineOptions::VendorPrimitiveDescriptor::Kind::Pool2D;
    vendor_descriptor.pool2d = desc;
    return materialize_apple_mps_vendor_program(
        module,
        plan,
        stage_type,
        vendor_descriptor,
        semantic_input_roles,
        gfx_mpsrt_make_external_buffer_abi_from_roles({GfxMpsrtExternalBufferRole::TensorInput,
                                                       GfxMpsrtExternalBufferRole::RuntimeParams,
                                                       GfxMpsrtExternalBufferRole::TensorOutput}),
        input_descs,
        output_descs);
}

GfxAppleProgramPipelineResult materialize_apple_mps_resize2d_program(
    mlir::ModuleOp module,
    const GfxStageOptimizationPlan& plan,
    const std::string& stage_type,
    const GfxMpsrtResize2DAbiDesc& desc,
    const std::vector<GfxKernelBufferRole>& semantic_input_roles,
    const std::vector<GfxMpsrtTensorDesc>& input_descs,
    const std::vector<GfxMpsrtTensorDesc>& output_descs) {
    GfxAppleStagePipelineOptions::VendorPrimitiveDescriptor vendor_descriptor{};
    vendor_descriptor.kind =
        GfxAppleStagePipelineOptions::VendorPrimitiveDescriptor::Kind::Resize2D;
    vendor_descriptor.resize2d = desc;
    return materialize_apple_mps_vendor_program(
        module,
        plan,
        stage_type,
        vendor_descriptor,
        semantic_input_roles,
        gfx_mpsrt_make_external_buffer_abi_from_roles({GfxMpsrtExternalBufferRole::TensorInput,
                                                       GfxMpsrtExternalBufferRole::TensorOutput}),
        input_descs,
        output_descs);
}

GfxAppleProgramPipelineResult materialize_apple_mps_softmax_program(
    mlir::ModuleOp module,
    const GfxStageOptimizationPlan& plan,
    const std::string& stage_type,
    const GfxMpsrtSoftmaxAbiDesc& desc,
    const std::vector<GfxKernelBufferRole>& semantic_input_roles,
    const std::vector<GfxMpsrtTensorDesc>& input_descs,
    const std::vector<GfxMpsrtTensorDesc>& output_descs) {
    GfxAppleStagePipelineOptions::VendorPrimitiveDescriptor vendor_descriptor{};
    vendor_descriptor.kind =
        GfxAppleStagePipelineOptions::VendorPrimitiveDescriptor::Kind::Softmax;
    vendor_descriptor.softmax = desc;
    const auto roles =
        semantic_input_roles.empty()
            ? std::vector<GfxKernelBufferRole>{GfxKernelBufferRole::TensorInput,
                                               GfxKernelBufferRole::TensorOutput,
                                               GfxKernelBufferRole::RuntimeParams}
            : semantic_input_roles;
    return materialize_apple_mps_vendor_program(
        module,
        plan,
        stage_type,
        vendor_descriptor,
        roles,
        gfx_mpsrt_make_external_buffer_abi_from_roles({GfxMpsrtExternalBufferRole::TensorInput,
                                                       GfxMpsrtExternalBufferRole::TensorOutput}),
        input_descs,
        output_descs);
}

GfxAppleProgramPipelineResult materialize_apple_mps_topk_program(
    mlir::ModuleOp module,
    const GfxStageOptimizationPlan& plan,
    const std::string& stage_type,
    const GfxMpsrtTopKAbiDesc& desc,
    const std::vector<GfxKernelBufferRole>& semantic_input_roles,
    const std::vector<GfxMpsrtTensorDesc>& input_descs,
    const std::vector<GfxMpsrtTensorDesc>& output_descs) {
    GfxAppleStagePipelineOptions::VendorPrimitiveDescriptor vendor_descriptor{};
    vendor_descriptor.kind =
        GfxAppleStagePipelineOptions::VendorPrimitiveDescriptor::Kind::TopK;
    vendor_descriptor.topk = desc;
    return materialize_apple_mps_vendor_program(
        module,
        plan,
        stage_type,
        vendor_descriptor,
        semantic_input_roles,
        gfx_mpsrt_make_external_buffer_abi_from_roles({GfxMpsrtExternalBufferRole::TensorInput,
                                                       GfxMpsrtExternalBufferRole::TensorOutput,
                                                       GfxMpsrtExternalBufferRole::TensorOutput}),
        input_descs,
        output_descs);
}

GfxAppleProgramPipelineResult materialize_apple_mpsrt_program(
    mlir::ModuleOp module,
    const GfxMpsrtProgram& program) {
    GfxAppleProgramPipelineResult result{};
    if (!module || !gfx_mpsrt_validate_program(program, nullptr)) {
        return result;
    }
    for (const auto& spec : program.stages) {
        if (!validate_apple_program_stage_contract(spec)) {
            return result;
        }
    }

    annotate_apple_program_pipeline(module, program);
    result.typed_program_materialized = materialize_module_mpsrt_ops(module, program);
    if (!result.typed_program_materialized) {
        result = {};
        return result;
    }
    if (!materialize_gfx_apple_mpsrt_runtime_abi_call_plan(module)) {
        result = {};
        return result;
    }
    result.valid = true;
    return result;
}

GfxAppleProgramPipelineResult materialize_apple_mpsrt_program_plan(
    mlir::ModuleOp module,
    const GfxAppleMpsrtProgramPlan& plan) {
    const auto program = make_apple_mpsrt_program_from_plan(plan);
    if (!program.valid) {
        return {};
    }
    return materialize_apple_mpsrt_program(module, program);
}

GfxAppleProgramPipelineResult materialize_apple_mps_gemm_program(
    mlir::ModuleOp module,
    const GfxAppleMpsGemmProgramDesc& desc) {
    if (!module || desc.record_key.empty()) {
        return {};
    }

    GfxAppleMpsrtProgramPlan plan{};
    plan.record_key = desc.record_key;
    plan.inputs = {desc.lhs, desc.rhs};
    plan.output_values = {2u};
    plan.stages.push_back({make_apple_mps_gemm_stage_desc(desc.gemm),
                           {0u, 1u},
                           {2u},
                           {desc.output}});
    plan.external_buffer_abi =
        gfx_mpsrt_make_external_buffer_abi_from_roles({GfxMpsrtExternalBufferRole::TensorInput,
                                                       GfxMpsrtExternalBufferRole::TensorInput,
                                                       GfxMpsrtExternalBufferRole::TensorOutput});

    return materialize_apple_mpsrt_program_plan(module, plan);
}

GfxAppleProgramPipelineResult materialize_apple_mps_gemm_msl_epilogue_program(
    mlir::ModuleOp module,
    const GfxAppleMpsGemmMslEpilogueProgramDesc& desc) {
    if (!module || desc.record_key.empty()) {
        return {};
    }

    GfxAppleMpsrtProgramPlan plan{};
    plan.record_key = desc.record_key;
    plan.inputs = {desc.lhs, desc.rhs};
    std::vector<GfxMpsrtExternalBufferRole> external_roles{
        GfxMpsrtExternalBufferRole::TensorInput,
        GfxMpsrtExternalBufferRole::TensorInput,
    };
    GfxMpsrtValue next_external_value = static_cast<GfxMpsrtValue>(plan.inputs.size());
    GfxMpsrtValue next_transient_value = next_external_value;
    GfxMpsrtValue bias_value = 0u;
    if (desc.has_bias) {
        bias_value = next_external_value++;
        next_transient_value = next_external_value;
        plan.inputs.push_back(desc.bias);
        external_roles.push_back(GfxMpsrtExternalBufferRole::TensorInput);
    }

    const auto gemm_output_value = next_transient_value++;
    const auto output_value = next_transient_value++;
    plan.stages.push_back({make_apple_mps_gemm_stage_desc(desc.gemm),
                           {0u, 1u},
                           {gemm_output_value},
                           {desc.gemm_output}});

    std::vector<GfxMpsrtValue> epilogue_inputs = {gemm_output_value};
    if (desc.has_bias) {
        epilogue_inputs.push_back(bias_value);
    }
    plan.stages.push_back({make_apple_msl_gemm_epilogue_stage_desc(desc.has_bias),
                           std::move(epilogue_inputs),
                           {output_value},
                           {desc.output}});
    plan.output_values = {output_value};
    external_roles.push_back(GfxMpsrtExternalBufferRole::TensorOutput);
    plan.external_buffer_abi =
        gfx_mpsrt_make_external_buffer_abi_from_roles(std::move(external_roles));

    return materialize_apple_mpsrt_program_plan(module, plan);
}

}  // namespace gfx_plugin
}  // namespace ov
