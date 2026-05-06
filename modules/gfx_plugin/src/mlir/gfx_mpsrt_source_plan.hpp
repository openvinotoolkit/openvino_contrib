// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "kernel_ir/gfx_codegen_backend.hpp"
#include "mlir/gfx_mpsrt_metadata.hpp"

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace ov {
namespace gfx_plugin {

enum class GfxMpsrtKernelSourcePlanKind {
    None,
    SingleStage,
    MultiStage,
};

struct GfxMpsrtKernelSourcePlan {
    GfxMpsrtKernelSourcePlanKind kind = GfxMpsrtKernelSourcePlanKind::None;
    KernelSource source;
    bool requires_mpsrt_model = false;
    std::string record_key;
    GfxMpsrtStageKind first_stage_kind = GfxMpsrtStageKind::Unknown;
    GfxMpsrtStageKind last_stage_kind = GfxMpsrtStageKind::Unknown;

    bool valid() const {
        return kind != GfxMpsrtKernelSourcePlanKind::None && source.module;
    }
};

namespace detail {

struct GfxMpsrtKernelSourceOptions {
    std::string msl_source;
    std::function<std::string(mlir::ModuleOp)> msl_generator;
    std::vector<uint32_t> spirv_binary;
    std::function<std::vector<uint32_t>(mlir::ModuleOp)> spirv_generator;
    uint32_t external_arg_count = 0;
    uint32_t external_output_arg_count = 0;
};

}  // namespace detail

inline bool gfx_mpsrt_stage_needs_custom_kernel_source(const GfxMpsrtStageDesc& stage) {
    return stage.kind == GfxMpsrtStageKind::MSLDispatch ||
           stage.kind == GfxMpsrtStageKind::SPIRVDispatch ||
           stage.uses_custom_kernel;
}

inline std::string gfx_mpsrt_stage_entry_point(const GfxMpsrtStageDesc& stage) {
    if (stage.stage_manifest.custom_kernel.valid &&
        !stage.stage_manifest.custom_kernel.entry_point.empty()) {
        return stage.stage_manifest.custom_kernel.entry_point;
    }
    if (!stage.dispatch_entry_point.empty()) {
        return stage.dispatch_entry_point;
    }
    if (!stage.kernel_name.empty()) {
        return stage.kernel_name;
    }
    return gfx_mpsrt_stage_kind_name(stage.kind);
}

namespace detail {

inline uint32_t gfx_mpsrt_source_plan_arg_count(const GfxMpsrtModuleBuilderPlan& module_plan,
                                                const GfxMpsrtKernelSourceOptions& options) {
    if (options.external_arg_count != 0) {
        return options.external_arg_count;
    }
    if (module_plan.builder_plan.external_buffer_count != 0) {
        return module_plan.builder_plan.external_buffer_count;
    }
    return static_cast<uint32_t>(module_plan.builder_plan.input_values.size() +
                                 module_plan.builder_plan.output_values.size());
}

inline uint32_t gfx_mpsrt_source_plan_output_arg_count(const GfxMpsrtModuleBuilderPlan& module_plan,
                                                       const GfxMpsrtKernelSourceOptions& options) {
    if (options.external_output_arg_count != 0) {
        return options.external_output_arg_count;
    }
    if (module_plan.builder_plan.external_output_buffer_count != 0) {
        return module_plan.builder_plan.external_output_buffer_count;
    }
    return static_cast<uint32_t>(module_plan.builder_plan.output_values.size());
}

inline GfxMpsrtKernelSourcePlan make_mpsrt_kernel_source_plan_from_module(
    mlir::ModuleOp module,
    GfxMpsrtKernelSourceOptions options) {
    GfxMpsrtKernelSourcePlan plan{};
    if (!module) {
        return plan;
    }

    GfxMpsrtModuleBuilderPlan module_plan;
    if (!build_module_mpsrt_builder_plan(module, module_plan)) {
        return plan;
    }

    const auto& program = module_plan.program;
    if (!program.valid || program.stages.empty()) {
        return plan;
    }

    const GfxMpsrtStageDesc* first_stage = nullptr;
    const GfxMpsrtStageDesc* last_stage = nullptr;
    const GfxMpsrtStageDesc* source_stage = nullptr;
    if (program.multi_stage) {
        first_stage = &program.stages.front().stage;
        last_stage = &program.stages.back().stage;
        for (auto it = program.stages.rbegin();
             it != program.stages.rend();
             ++it) {
            if (gfx_mpsrt_stage_needs_custom_kernel_source(it->stage)) {
                source_stage = &it->stage;
                break;
            }
        }
        if (!source_stage) {
            source_stage = last_stage;
        }
        plan.kind = GfxMpsrtKernelSourcePlanKind::MultiStage;
        plan.record_key = program.record_key;
    } else {
        first_stage = &module_plan.stage_plan.stage;
        last_stage = first_stage;
        source_stage = first_stage;
        plan.kind = GfxMpsrtKernelSourcePlanKind::SingleStage;
        plan.record_key = module_plan.stage_plan.stage_record_key;
    }

    if (!first_stage || !last_stage || !source_stage) {
        return {};
    }

    plan.first_stage_kind = first_stage->kind;
    plan.last_stage_kind = last_stage->kind;
    plan.requires_mpsrt_model = true;
    plan.source.module = module;
    plan.source.entry_point = gfx_mpsrt_stage_entry_point(*source_stage);
    plan.source.msl_source = std::move(options.msl_source);
    plan.source.msl_generator = std::move(options.msl_generator);
    plan.source.spirv_binary = std::move(options.spirv_binary);
    plan.source.spirv_generator = std::move(options.spirv_generator);
    plan.source.signature.arg_count = gfx_mpsrt_source_plan_arg_count(module_plan, options);
    plan.source.signature.output_arg_count = gfx_mpsrt_source_plan_output_arg_count(module_plan, options);
    return plan;
}

}  // namespace detail

inline GfxMpsrtKernelSourcePlan make_mpsrt_kernel_source_plan_from_module(
    mlir::ModuleOp module) {
    return detail::make_mpsrt_kernel_source_plan_from_module(module, detail::GfxMpsrtKernelSourceOptions{});
}

inline GfxMpsrtKernelSourcePlan make_mpsrt_kernel_source_plan_from_msl_source(
    mlir::ModuleOp module,
    std::string msl_source) {
    detail::GfxMpsrtKernelSourceOptions options{};
    options.msl_source = std::move(msl_source);
    return detail::make_mpsrt_kernel_source_plan_from_module(module, std::move(options));
}

inline GfxMpsrtKernelSourcePlan make_mpsrt_kernel_source_plan_from_msl_generator(
    mlir::ModuleOp module,
    std::function<std::string(mlir::ModuleOp)> msl_generator) {
    detail::GfxMpsrtKernelSourceOptions options{};
    options.msl_generator = std::move(msl_generator);
    return detail::make_mpsrt_kernel_source_plan_from_module(module, std::move(options));
}

inline GfxMpsrtKernelSourcePlan make_mpsrt_kernel_source_plan_from_configured_source(
    KernelSource source) {
    detail::GfxMpsrtKernelSourceOptions options{};
    options.msl_source = std::move(source.msl_source);
    options.msl_generator = std::move(source.msl_generator);
    options.spirv_binary = std::move(source.spirv_binary);
    options.spirv_generator = std::move(source.spirv_generator);
    options.external_arg_count = source.signature.arg_count;
    options.external_output_arg_count = source.signature.output_arg_count;
    return detail::make_mpsrt_kernel_source_plan_from_module(source.module, std::move(options));
}

}  // namespace gfx_plugin
}  // namespace ov
