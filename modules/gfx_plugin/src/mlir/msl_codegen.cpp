// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen.hpp"

#include "mlir/gfx_mpsrt_metadata.hpp"

#include "mlir/IR/Builders.h"

#include "llvm/ADT/SmallVector.h"

#include <utility>
#include <vector>

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

bool read_positive_i32_attr(mlir::ModuleOp module, const char* name, uint32_t& out) {
    auto attr = module->getAttrOfType<mlir::IntegerAttr>(name);
    if (!attr || attr.getInt() <= 0) {
        return false;
    }
    out = static_cast<uint32_t>(attr.getInt());
    return true;
}

bool module_has_mpsrt_external_buffer_abi_attrs(mlir::ModuleOp module) {
    return module->hasAttr("gfx.mpsrt.external_buffer_count") ||
           module->hasAttr("gfx.mpsrt.external_output_buffer_count") ||
           module->hasAttr("gfx.mpsrt.external_buffer_roles");
}

void annotate_mpsrt_external_buffer_roles(mlir::ModuleOp module,
                                          const std::vector<GfxMpsrtExternalBufferRole>& roles) {
    mlir::Builder builder(module.getContext());

    uint32_t output_buffer_count = 0;
    llvm::SmallVector<mlir::Attribute, 8> role_attrs;
    role_attrs.reserve(roles.size());
    for (const auto role : roles) {
        if (role == GfxMpsrtExternalBufferRole::TensorOutput) {
            ++output_buffer_count;
        }
        role_attrs.push_back(builder.getI32IntegerAttr(static_cast<int32_t>(role)));
    }

    module->setAttr("gfx.mpsrt.external_buffer_count",
                    builder.getI32IntegerAttr(static_cast<int32_t>(roles.size())));
    module->setAttr("gfx.mpsrt.external_output_buffer_count",
                    builder.getI32IntegerAttr(static_cast<int32_t>(output_buffer_count)));
    module->setAttr("gfx.mpsrt.external_buffer_roles", builder.getArrayAttr(role_attrs));
}

std::vector<GfxMpsrtExternalBufferRole> make_roles_from_leading_io_spec(
    const GfxMslExternalBufferAbiSpec& spec,
    uint32_t buffer_count) {
    const uint32_t structured_count = spec.leading_input_count + spec.leading_output_count;
    if (!spec.valid || structured_count == 0 || buffer_count < structured_count) {
        return {};
    }

    std::vector<GfxMpsrtExternalBufferRole> roles;
    roles.reserve(buffer_count);
    roles.insert(roles.end(),
                 spec.leading_input_count,
                 GfxMpsrtExternalBufferRole::TensorInput);
    roles.insert(roles.end(),
                 spec.leading_output_count,
                 GfxMpsrtExternalBufferRole::TensorOutput);
    roles.insert(roles.end(),
                 buffer_count - structured_count,
                 GfxMpsrtExternalBufferRole::RuntimeParams);
    return roles;
}

bool annotate_msl_role_based_external_buffer_abi(mlir::ModuleOp module,
                                                 const GfxMslKernelPlan& msl_plan,
                                                 uint32_t known_buffer_count = 0) {
    if (!module || module_has_mpsrt_external_buffer_abi_attrs(module)) {
        return false;
    }
    if (!msl_plan.external_buffer_abi.valid) {
        return false;
    }
    std::vector<GfxMpsrtExternalBufferRole> roles = msl_plan.external_buffer_abi.roles;
    if (roles.empty()) {
        uint32_t buffer_count = known_buffer_count;
        if (buffer_count == 0) {
            (void)read_positive_i32_attr(module, "gfx.fixed_arg_count", buffer_count);
        }
        roles = make_roles_from_leading_io_spec(msl_plan.external_buffer_abi, buffer_count);
    }
    if (roles.empty()) {
        return false;
    }
    annotate_mpsrt_external_buffer_roles(module, roles);
    return true;
}

void annotate_msl_tail_output_external_buffer_abi(mlir::ModuleOp module,
                                                  const GfxMslKernelPlan& msl_plan) {
    if (!module ||
        !msl_plan.external_buffer_abi.valid ||
        !msl_plan.external_buffer_abi.tail_outputs) {
        return;
    }
    if (module_has_mpsrt_external_buffer_abi_attrs(module)) {
        return;
    }

    uint32_t buffer_count = 0;
    uint32_t output_buffer_count = 0;
    if (!read_positive_i32_attr(module, "gfx.fixed_arg_count", buffer_count) ||
        !read_positive_i32_attr(module, "gfx.kernel_output_arg_count", output_buffer_count) ||
        output_buffer_count > buffer_count) {
        return;
    }

    std::vector<GfxMpsrtExternalBufferRole> roles;
    roles.reserve(buffer_count);
    const uint32_t input_buffer_count = buffer_count - output_buffer_count;
    for (uint32_t i = 0; i < buffer_count; ++i) {
        roles.push_back(i < input_buffer_count
                            ? GfxMpsrtExternalBufferRole::TensorInput
                            : GfxMpsrtExternalBufferRole::TensorOutput);
    }
    annotate_mpsrt_external_buffer_roles(module, roles);
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
    (void)annotate_msl_role_based_external_buffer_abi(source.module,
                                                     msl_plan,
                                                     source.signature.arg_count);
}

void annotate_msl_module_with_stage_plan(mlir::ModuleOp module,
                                         const GfxStageOptimizationPlan& plan,
                                         const std::string& stage_type) {
    if (!module) {
        return;
    }

    annotate_module_with_mpsrt_stage_plan(module, plan, stage_type);

    GfxMpsrtModuleStagePlan stage_plan;
    if (!read_module_mpsrt_stage_plan(module, stage_plan) ||
        stage_plan.stage.kind != GfxMpsrtStageKind::MSLDispatch) {
        return;
    }

    const auto msl_plan = make_msl_kernel_plan(stage_type, stage_plan.stage.kernel_name);
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
    module->setAttr("gfx.mpsrt.dispatch_kernel_family", builder.getStringAttr(msl_plan.family_name));
    module->setAttr("gfx.mpsrt.dispatch_entry_point", builder.getStringAttr(msl_plan.required_entry_point));
    module->setAttr("gfx.mpsrt.dispatch_kernel_family_id",
                    builder.getI32IntegerAttr(static_cast<int32_t>(msl_plan.abi_kernel_family)));
    module->setAttr("gfx.mpsrt.dispatch_flags",
                    builder.getI32IntegerAttr(static_cast<int32_t>(msl_plan.dispatch_flags)));
    module->setAttr("gfx.mpsrt.dispatch_precompiled_kernel_required",
                    builder.getBoolAttr(msl_plan.precompiled_metallib_required));
    module->setAttr("gfx.mpsrt.dispatch_threads_per_threadgroup",
                    builder.getI32IntegerAttr(static_cast<int32_t>(msl_plan.threads_per_threadgroup)));
    if (!annotate_msl_role_based_external_buffer_abi(module, msl_plan)) {
        annotate_msl_tail_output_external_buffer_abi(module, msl_plan);
    }
}

}  // namespace gfx_plugin
}  // namespace ov
