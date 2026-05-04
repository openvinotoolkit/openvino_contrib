// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/gfx_apple_stage_pipeline.hpp"

#include <utility>

#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace ov {
namespace gfx_plugin {
namespace {

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
        if (!annotate_module_with_mpsrt_stage_manifest(getOperation(),
                                                       m_options.plan,
                                                       m_options.stage_type,
                                                       m_options.kernel_entry_point,
                                                       stage_plan)) {
            getOperation()->emitError("failed to materialize Apple stage manifest for ")
                << m_options.stage_type;
            signalPassFailure();
        }
    }

private:
    GfxAppleStagePipelineOptions m_options;
};

}  // namespace

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

std::unique_ptr<mlir::Pass> createGfxAppleStageManifestPass(
    const GfxAppleStagePipelineOptions& options) {
    return std::make_unique<GfxAppleStageManifestPass>(options);
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
    pm.addPass(createGfxAppleStageManifestPass(options));
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    if (mlir::failed(pm.run(module))) {
        return result;
    }

    if (!build_module_mpsrt_stage_plan(module,
                                       options.plan,
                                       options.stage_type,
                                       options.kernel_entry_point,
                                       result.stage_plan)) {
        return result;
    }
    result.valid = true;

    if (options.materialize_typed_program) {
        GfxMpsrtExternalBufferAbiPlan external_buffer_abi{};
        if (result.stage_plan.stage.uses_custom_kernel) {
            (void)gfx_mpsrt_external_buffer_abi_from_kernel_manifest(module,
                                                                    external_buffer_abi);
        }
        result.typed_program_materialized =
            materialize_module_mpsrt_ops_from_stage_plan(module,
                                                         result.stage_plan,
                                                         external_buffer_abi);
        if (!result.typed_program_materialized) {
            result = {};
            return result;
        }
    }
    return result;
}

}  // namespace gfx_plugin
}  // namespace ov
