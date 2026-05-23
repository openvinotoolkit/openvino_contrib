// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/gfx_apple_stage_pipeline.hpp"

#include <algorithm>
#include <utility>

#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "runtime/gfx_mpsrt_program.hpp"

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
    }
    return passes;
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
            case GfxKernelBackendDomain::OpenCl:
                stage.domain = GfxStageBackendDomain::OpenCl;
                break;
            case GfxKernelBackendDomain::Spirv:
                stage.domain = GfxStageBackendDomain::Spirv;
                break;
            case GfxKernelBackendDomain::Unknown:
            default:
                break;
        }
    }
    if (stage.kind == GfxMpsrtStageKind::MSLDispatch) {
        const auto dispatch =
            gfx_mpsrt_custom_dispatch_spec_from_kernel_manifest(stage.stage_manifest.custom_kernel);
        if (dispatch.valid) {
            stage.kernel_name = dispatch.entry_point;
        }
    }
    if (stage.kernel_name.empty()) {
        stage.kernel_name = gfx_mpsrt_stage_default_kernel_name(stage);
    }
    return stage.kind != GfxMpsrtStageKind::Unknown &&
           stage.domain != GfxStageBackendDomain::Unknown &&
           gfx_mpsrt_stage_has_builder_symbol(stage.kind);
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
        if (gfx_mpsrt_stage_record_key(stage).empty()) {
            program = {};
            return program;
        }
        program.stages.push_back({std::move(stage),
                                  request.inputs,
                                  request.outputs,
                                  request.output_descs});
    }

    if (program.output_values.empty()) {
        program.output_values = program.stages.back().outputs;
    }

    if (!program.external_buffer_abi.valid) {
        program.external_buffer_abi = gfx_mpsrt_external_buffer_abi_from_typed_vendor_program_io(program);
        if (!program.external_buffer_abi.valid) {
            program = {};
            return program;
        }
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

bool apple_vendor_descriptor_matches_stage(
    GfxMpsrtStageKind stage_kind,
    GfxAppleMpsVendorPrimitiveKind descriptor_kind) {
    using Kind = GfxAppleMpsVendorPrimitiveKind;
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
        case Kind::Sdpa:
            return stage_kind == GfxMpsrtStageKind::MPSSdpa;
    }
    return false;
}

bool apply_apple_vendor_descriptor(
    GfxMpsrtModuleStagePlan& stage_plan,
    const GfxAppleMpsVendorPrimitiveDescriptor& descriptor) {
    using Kind = GfxAppleMpsVendorPrimitiveKind;
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
        case Kind::Sdpa:
            stage_plan.stage.sdpa_desc = descriptor.sdpa;
            break;
        case Kind::None:
            break;
    }

    return finalize_mpsrt_module_stage_plan(stage_plan);
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
        if (stage_plan.inputs.size() != options.input_descs.size() &&
            options.vendor_descriptor.kind == GfxAppleMpsVendorPrimitiveKind::None) {
            return false;
        }
        stage_plan.inputs = options.input_descs;
    }
    if (!options.output_descs.empty()) {
        if (stage_plan.outputs.size() != options.output_descs.size() &&
            options.vendor_descriptor.kind == GfxAppleMpsVendorPrimitiveKind::None) {
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
    if (stage_plan.stage.stage_manifest.valid) {
        stage_plan.stage.stage_manifest.semantic_input_roles =
            detail::gfx_mpsrt_normalize_semantic_input_roles(stage_plan.inputs.size(),
                                                             options.semantic_input_roles);
        stage_plan.stage.stage_manifest.semantic_output_roles =
            detail::gfx_mpsrt_default_semantic_output_roles(stage_plan.outputs.size());
    }
    return finalize_mpsrt_module_stage_plan(stage_plan);
}

GfxAppleProgramPipelineResult materialize_apple_mps_vendor_program(
    mlir::ModuleOp module,
    const GfxStageOptimizationPlan& plan,
    const std::string& stage_type,
    const GfxAppleMpsVendorPrimitiveDescriptor& vendor_descriptor,
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
    program_plan.record_key = gfx_mpsrt_stage_plan_record_key(stage_pipeline.stage_plan);
    program_plan.inputs = stage_pipeline.stage_plan.inputs;
    program_plan.output_values = output_values;
    program_plan.external_buffer_abi = external_buffer_abi;
    program_plan.stages.push_back({stage_pipeline.stage_plan.stage,
                                   input_values,
                                   output_values,
                                   stage_pipeline.stage_plan.outputs});
    if (build_mpsrt_storage_assignment_from_stage_plan(stage_pipeline.stage_plan,
                                                       program_plan.storage_bridges)) {
        program_plan.has_storage_bridges = true;
    }

    return materialize_apple_mpsrt_program_plan(module, program_plan);
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
        return "Validate Apple backend placement before stage manifest lowering";
    }

    void runOnOperation() final {
        auto module = getOperation();
        const auto& placement = m_options.plan.placement;
        if (m_options.stage_type.empty() ||
            placement.domain == GfxStageBackendDomain::Unknown ||
            placement.storage == GfxStageStorageKind::Unknown) {
            module->emitError("invalid Apple stage placement for ")
                << m_options.stage_type;
            signalPassFailure();
        }
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
        return "Assign Apple storage bridge contract before stage manifest lowering";
    }

    void runOnOperation() final {
        auto module = getOperation();
        GfxMpsrtModuleStagePlan stage_plan{};
        if (!build_module_mpsrt_stage_plan(module,
                                           m_options.plan,
                                           m_options.stage_type,
                                           m_options.kernel_entry_point,
                                           stage_plan,
                                           m_options.semantic_input_roles)) {
            module->emitError("failed to build Apple stage plan for ")
                << m_options.stage_type;
            signalPassFailure();
            return;
        }
        if (!apply_apple_stage_tensor_desc_overrides(stage_plan, m_options)) {
            module->emitError("failed to apply Apple tensor descriptor contract for ")
                << m_options.stage_type;
            signalPassFailure();
            return;
        }
        if (stage_plan.stage.input_storage == GfxMpsrtStorage::Buffer &&
            stage_plan.stage.output_storage == GfxMpsrtStorage::Buffer) {
            return;
        }
        if (stage_plan.stage.input_storage != GfxMpsrtStorage::Image &&
            stage_plan.stage.output_storage != GfxMpsrtStorage::Image) {
            return;
        }
        std::vector<GfxMpsrtStorageBridgeDesc> bridges;
        if (!build_mpsrt_storage_assignment_from_stage_plan(stage_plan, bridges)) {
            module->emitError("failed to build Apple storage bridge plan for ")
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
        return "Reserve Apple post-op fusion boundary before stage manifest lowering";
    }

    void runOnOperation() final {}

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
        if (m_options.vendor_descriptor.kind ==
            GfxAppleMpsVendorPrimitiveKind::None) {
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
        if (stage_plan.stage.stage_manifest.execution_kind == GfxKernelExecutionKind::CustomKernel) {
            (void)gfx_mpsrt_external_buffer_abi_from_kernel_manifest(module,
                                                                    external_buffer_abi);
        }
        if (!materialize_module_mpsrt_ops_from_stage_plan(module,
                                                          stage_plan,
                                                          external_buffer_abi)) {
            signalPassFailure();
        }
    }

private:
    GfxAppleStagePipelineOptions m_options;
};

}  // namespace

GfxAppleMpsrtProgramPlanBuilder::GfxAppleMpsrtProgramPlanBuilder(std::string record_key) {
    plan_.record_key = std::move(record_key);
}

GfxMpsrtValue GfxAppleMpsrtProgramPlanBuilder::add_external_input(
    GfxMpsrtTensorDesc desc,
    GfxMpsrtExternalBufferRole role) {
    if (!valid_ || !plan_.stages.empty() || next_value_ != plan_.inputs.size()) {
        valid_ = false;
        return 0u;
    }
    const auto value = next_value_++;
    plan_.inputs.push_back(std::move(desc));
    external_roles_.push_back(role);
    return value;
}

std::vector<GfxMpsrtValue> GfxAppleMpsrtProgramPlanBuilder::add_stage(
    GfxMpsrtStageDesc stage,
    std::vector<GfxMpsrtValue> inputs,
    std::vector<GfxMpsrtTensorDesc> output_descs) {
    std::vector<GfxMpsrtValue> outputs;
    if (!valid_ || inputs.empty() || output_descs.empty()) {
        valid_ = false;
        return outputs;
    }
    outputs.reserve(output_descs.size());
    for (size_t i = 0; i < output_descs.size(); ++i) {
        outputs.push_back(next_value_++);
    }
    plan_.stages.push_back({std::move(stage),
                            std::move(inputs),
                            outputs,
                            std::move(output_descs)});
    return outputs;
}

GfxMpsrtValue GfxAppleMpsrtProgramPlanBuilder::add_single_output_stage(
    GfxMpsrtStageDesc stage,
    std::vector<GfxMpsrtValue> inputs,
    GfxMpsrtTensorDesc output_desc) {
    auto outputs = add_stage(std::move(stage),
                             std::move(inputs),
                             {std::move(output_desc)});
    if (outputs.size() != 1) {
        valid_ = false;
        return 0u;
    }
    return outputs.front();
}

void GfxAppleMpsrtProgramPlanBuilder::add_external_output(
    GfxMpsrtValue value,
    GfxMpsrtExternalBufferRole role) {
    if (!valid_) {
        return;
    }
    plan_.output_values.push_back(value);
    external_roles_.push_back(role);
}

void GfxAppleMpsrtProgramPlanBuilder::set_storage_bridges(
    std::vector<GfxMpsrtStorageBridgeDesc> storage_bridges) {
    if (!valid_) {
        return;
    }
    plan_.has_storage_bridges = !storage_bridges.empty();
    plan_.storage_bridges = std::move(storage_bridges);
}

GfxAppleMpsrtProgramPlan GfxAppleMpsrtProgramPlanBuilder::finalize() const {
    if (!valid_ || plan_.record_key.empty() || plan_.stages.empty()) {
        return {};
    }

    auto plan = plan_;
    if (plan.output_values.empty()) {
        plan.output_values = plan.stages.back().outputs;
    }
    if (plan.output_values.empty() || external_roles_.empty()) {
        return {};
    }
    plan.external_buffer_abi =
        gfx_mpsrt_make_external_buffer_abi_from_roles(external_roles_);
    return plan;
}

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
    }
    return result;
}

GfxAppleProgramPipelineResult materialize_apple_mps_vendor_contract_program(
    mlir::ModuleOp module,
    const GfxStageOptimizationPlan& plan,
    const std::string& stage_type,
    const GfxAppleMpsVendorPrimitiveContract& contract) {
    if (!contract.valid) {
        return {};
    }
    return materialize_apple_mps_vendor_program(module,
                                                plan,
                                                stage_type,
                                                contract.descriptor,
                                                contract.semantic_input_roles,
                                                contract.external_buffer_abi,
                                                contract.input_descs,
                                                contract.output_descs);
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

    result.typed_program_materialized = materialize_module_mpsrt_ops(module, program);
    if (!result.typed_program_materialized) {
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

}  // namespace gfx_plugin
}  // namespace ov
