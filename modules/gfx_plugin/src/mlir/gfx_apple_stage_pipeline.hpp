// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/gfx_apple_vendor_descriptors.hpp"
#include "mlir/gfx_mpsrt_metadata.hpp"
#include "runtime/gfx_stage_policy.hpp"

namespace ov {
namespace gfx_plugin {

struct GfxAppleStagePipelineOptions {
    GfxStageOptimizationPlan plan;
    std::string stage_type;
    std::string kernel_entry_point;
    std::vector<GfxKernelBufferRole> semantic_input_roles;
    std::vector<GfxMpsrtTensorDesc> input_descs;
    std::vector<GfxMpsrtTensorDesc> output_descs;
    GfxAppleMpsVendorPrimitiveDescriptor vendor_descriptor;
    bool materialize_typed_program = true;
};

struct GfxAppleStagePipelineResult {
    bool valid = false;
    bool typed_program_materialized = false;
    GfxMpsrtModuleStagePlan stage_plan;
};

struct GfxAppleProgramPipelineResult {
    bool valid = false;
    bool typed_program_materialized = false;
};

struct GfxAppleMpsrtProgramStageRequest {
    GfxMpsrtStageDesc stage;
    std::vector<GfxMpsrtValue> inputs;
    std::vector<GfxMpsrtValue> outputs;
    std::vector<GfxMpsrtTensorDesc> output_descs;
};

struct GfxAppleMpsrtProgramPlan {
    std::string record_key;
    std::vector<GfxMpsrtTensorDesc> inputs;
    std::vector<GfxMpsrtValue> output_values;
    std::vector<GfxAppleMpsrtProgramStageRequest> stages;
    GfxMpsrtExternalBufferAbiPlan external_buffer_abi{};
    bool has_storage_bridges = false;
    std::vector<GfxMpsrtStorageBridgeDesc> storage_bridges;
};

class GfxAppleMpsrtProgramPlanBuilder {
public:
    explicit GfxAppleMpsrtProgramPlanBuilder(std::string record_key);

    GfxMpsrtValue add_external_input(
        GfxMpsrtTensorDesc desc,
        GfxMpsrtExternalBufferRole role = GfxMpsrtExternalBufferRole::TensorInput);

    std::vector<GfxMpsrtValue> add_stage(
        GfxMpsrtStageDesc stage,
        std::vector<GfxMpsrtValue> inputs,
        std::vector<GfxMpsrtTensorDesc> output_descs);

    GfxMpsrtValue add_single_output_stage(
        GfxMpsrtStageDesc stage,
        std::vector<GfxMpsrtValue> inputs,
        GfxMpsrtTensorDesc output_desc);

    void add_external_output(
        GfxMpsrtValue value,
        GfxMpsrtExternalBufferRole role = GfxMpsrtExternalBufferRole::TensorOutput);

    void set_storage_bridges(std::vector<GfxMpsrtStorageBridgeDesc> storage_bridges);

    GfxAppleMpsrtProgramPlan finalize() const;

private:
    GfxAppleMpsrtProgramPlan plan_;
    std::vector<GfxMpsrtExternalBufferRole> external_roles_;
    GfxMpsrtValue next_value_ = 0;
    bool valid_ = true;
};

enum class GfxAppleStagePipelinePassKind {
    CoreCanonicalize,
    Placement,
    StorageAssignment,
    Fusion,
    VendorDescriptor,
    StageManifest,
    RuntimeAbi,
    RuntimeAbiCallPlan,
};

const char* gfx_apple_stage_pipeline_pass_name(GfxAppleStagePipelinePassKind kind);
std::vector<GfxAppleStagePipelinePassKind> gfx_apple_stage_pipeline_pass_boundaries(
    bool materialize_typed_program);

std::unique_ptr<mlir::Pass> createGfxAppleCanonicalizePass();
std::unique_ptr<mlir::Pass> createGfxApplePlacementPass(
    const GfxAppleStagePipelineOptions& options);
std::unique_ptr<mlir::Pass> createGfxAppleStorageAssignmentPass(
    const GfxAppleStagePipelineOptions& options);
std::unique_ptr<mlir::Pass> createGfxAppleFusionPass(
    const GfxAppleStagePipelineOptions& options);
std::unique_ptr<mlir::Pass> createGfxAppleVendorDescriptorPass(
    const GfxAppleStagePipelineOptions& options);
std::unique_ptr<mlir::Pass> createGfxAppleStageManifestPass(
    const GfxAppleStagePipelineOptions& options);
std::unique_ptr<mlir::Pass> createGfxAppleRuntimeAbiPass(
    const GfxAppleStagePipelineOptions& options);

GfxAppleStagePipelineResult run_gfx_apple_stage_pipeline(
    mlir::ModuleOp module,
    const GfxAppleStagePipelineOptions& options);

GfxAppleProgramPipelineResult materialize_apple_mps_vendor_contract_program(
    mlir::ModuleOp module,
    const GfxStageOptimizationPlan& plan,
    const std::string& stage_type,
    const GfxAppleMpsVendorPrimitiveContract& contract);

GfxAppleProgramPipelineResult materialize_apple_mpsrt_program(
    mlir::ModuleOp module,
    const GfxMpsrtProgram& program);

GfxAppleProgramPipelineResult materialize_apple_mpsrt_program_plan(
    mlir::ModuleOp module,
    const GfxAppleMpsrtProgramPlan& plan);

inline GfxAppleStagePipelineResult run_gfx_apple_stage_pipeline(
    mlir::ModuleOp module,
    const GfxStageOptimizationPlan& plan,
    std::string_view stage_type,
    std::string_view kernel_entry_point = {},
    bool materialize_typed_program = true) {
    GfxAppleStagePipelineOptions options{};
    options.plan = plan;
    options.stage_type = std::string(stage_type);
    options.kernel_entry_point = std::string(kernel_entry_point);
    options.materialize_typed_program = materialize_typed_program;
    return run_gfx_apple_stage_pipeline(module, options);
}

}  // namespace gfx_plugin
}  // namespace ov
