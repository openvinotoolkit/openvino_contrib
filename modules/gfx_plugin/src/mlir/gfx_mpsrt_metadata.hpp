// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/gfx_mpsrt_ops.hpp"
#include "openvino/core/type/element_type.hpp"
#include "runtime/gfx_mpsrt_abi.hpp"
#include "runtime/gfx_mpsrt_builder_plan.hpp"
#include "runtime/gfx_mpsrt_kernel_manifest_adapter.hpp"
#include "runtime/gfx_mpsrt_plan.hpp"
#include "runtime/gfx_mpsrt_program.hpp"
#include "runtime/gfx_stage_policy.hpp"

namespace ov {
namespace gfx_plugin {

namespace detail {

inline mlir::func::FuncOp gfx_mpsrt_entry_func(mlir::ModuleOp module) {
    if (!module) {
        return {};
    }
    for (auto func : module.getOps<mlir::func::FuncOp>()) {
        return func;
    }
    return {};
}

inline ov::element::Type gfx_mpsrt_element_from_mlir_type(mlir::Type type) {
    if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type)) {
        type = shaped.getElementType();
    }
    if (mlir::isa<mlir::Float16Type>(type)) {
        return ov::element::f16;
    }
    if (mlir::isa<mlir::Float32Type>(type)) {
        return ov::element::f32;
    }
    if (mlir::isa<mlir::IndexType>(type)) {
        return ov::element::i64;
    }
    if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type)) {
        if (integer.getWidth() == 1) {
            return ov::element::boolean;
        }
        const bool is_unsigned = integer.isUnsigned();
        switch (integer.getWidth()) {
            case 8:
                return is_unsigned ? ov::element::u8 : ov::element::i8;
            case 32:
                return is_unsigned ? ov::element::u32 : ov::element::i32;
            case 64:
                return is_unsigned ? ov::element::u64 : ov::element::i64;
            default:
                return ov::element::dynamic;
        }
    }
    return ov::element::dynamic;
}

inline std::vector<int64_t> gfx_mpsrt_shape_from_mlir_type(mlir::Type type) {
    if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type)) {
        if (!shaped.hasRank()) {
            return {};
        }
        return std::vector<int64_t>(shaped.getShape().begin(), shaped.getShape().end());
    }
    return {};
}

inline mlir::ArrayAttr gfx_mpsrt_i64_array_attr(mlir::Builder& builder,
                                                const uint64_t* values,
                                                uint32_t count) {
    llvm::SmallVector<mlir::Attribute, 8> attrs;
    attrs.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        attrs.push_back(builder.getI64IntegerAttr(static_cast<int64_t>(values[i])));
    }
    return builder.getArrayAttr(attrs);
}

inline mlir::ArrayAttr gfx_mpsrt_i64_array_attr(mlir::Builder& builder,
                                                const int64_t* values,
                                                uint32_t count) {
    llvm::SmallVector<mlir::Attribute, 8> attrs;
    attrs.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        attrs.push_back(builder.getI64IntegerAttr(values[i]));
    }
    return builder.getArrayAttr(attrs);
}

inline GfxStageBackendDomain gfx_mpsrt_backend_domain_from_name(llvm::StringRef name) {
    if (name == "apple_mps") return GfxStageBackendDomain::AppleMps;
    if (name == "apple_msl") return GfxStageBackendDomain::AppleMsl;
    if (name == "spirv") return GfxStageBackendDomain::Spirv;
    return GfxStageBackendDomain::Unknown;
}

inline bool gfx_mpsrt_read_i32_attr(mlir::Operation* module, const std::string& name, uint32_t& out) {
    auto attr = module->getAttrOfType<mlir::IntegerAttr>(name);
    if (!attr) {
        return false;
    }
    out = static_cast<uint32_t>(attr.getInt());
    return true;
}

inline bool gfx_mpsrt_read_i64_attr(mlir::Operation* module, const std::string& name, uint64_t& out) {
    auto attr = module->getAttrOfType<mlir::IntegerAttr>(name);
    if (!attr) {
        return false;
    }
    out = static_cast<uint64_t>(attr.getInt());
    return true;
}

inline bool gfx_mpsrt_read_string_attr(mlir::Operation* module, const std::string& name, std::string& out) {
    auto attr = module->getAttrOfType<mlir::StringAttr>(name);
    if (!attr) {
        return false;
    }
    out = attr.str();
    return true;
}

inline bool gfx_mpsrt_read_bool_attr(mlir::Operation* module, const std::string& name, bool& out) {
    auto attr = module->getAttrOfType<mlir::BoolAttr>(name);
    if (!attr) {
        return false;
    }
    out = attr.getValue();
    return true;
}

inline void gfx_mpsrt_read_u64_array_attr(mlir::Operation* module,
                                          const std::string& name,
                                          uint64_t* out,
                                          uint32_t count) {
    auto attr = module->getAttrOfType<mlir::ArrayAttr>(name);
    if (!attr) {
        return;
    }
    const uint32_t n = std::min<uint32_t>(count, static_cast<uint32_t>(attr.size()));
    for (uint32_t i = 0; i < n; ++i) {
        if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(attr[i])) {
            out[i] = static_cast<uint64_t>(int_attr.getInt());
        }
    }
}

inline void gfx_mpsrt_read_i64_array_attr(mlir::Operation* module,
                                          const std::string& name,
                                          int64_t* out,
                                          uint32_t count) {
    auto attr = module->getAttrOfType<mlir::ArrayAttr>(name);
    if (!attr) {
        return;
    }
    const uint32_t n = std::min<uint32_t>(count, static_cast<uint32_t>(attr.size()));
    for (uint32_t i = 0; i < n; ++i) {
        if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(attr[i])) {
            out[i] = int_attr.getInt();
        }
    }
}

inline std::vector<uint32_t> gfx_mpsrt_read_u32_vector_attr(mlir::Operation* module,
                                                            const std::string& name) {
    std::vector<uint32_t> values;
    auto attr = module->getAttrOfType<mlir::ArrayAttr>(name);
    if (!attr) {
        return values;
    }
    values.reserve(attr.size());
    for (auto value_attr : attr) {
        if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(value_attr)) {
            values.push_back(static_cast<uint32_t>(int_attr.getInt()));
        }
    }
    return values;
}

inline mlir::ArrayAttr gfx_mpsrt_u32_vector_attr(mlir::Builder& builder,
                                                 const std::vector<uint32_t>& values) {
    llvm::SmallVector<mlir::Attribute, 8> attrs;
    attrs.reserve(values.size());
    for (const auto value : values) {
        attrs.push_back(builder.getI32IntegerAttr(static_cast<int32_t>(value)));
    }
    return builder.getArrayAttr(attrs);
}

inline void gfx_mpsrt_set_external_buffer_roles_attrs(
    mlir::Operation* module,
    const std::string& prefix,
    const std::vector<GfxMpsrtExternalBufferRole>& roles) {
    mlir::Builder builder(module->getContext());

    uint32_t output_buffer_count = 0;
    llvm::SmallVector<mlir::Attribute, 8> role_attrs;
    role_attrs.reserve(roles.size());
    for (const auto role : roles) {
        if (role == GfxMpsrtExternalBufferRole::TensorOutput) {
            ++output_buffer_count;
        }
        role_attrs.push_back(builder.getI32IntegerAttr(static_cast<int32_t>(role)));
    }

    module->setAttr(prefix + ".external_buffer_count",
                    builder.getI32IntegerAttr(static_cast<int32_t>(roles.size())));
    module->setAttr(prefix + ".external_output_buffer_count",
                    builder.getI32IntegerAttr(static_cast<int32_t>(output_buffer_count)));
    module->setAttr(prefix + ".external_buffer_roles", builder.getArrayAttr(role_attrs));
}

inline std::vector<uint32_t> gfx_kernel_buffer_roles_to_attr_values(
    const std::vector<GfxKernelBufferRole>& roles) {
    std::vector<uint32_t> values;
    values.reserve(roles.size());
    for (const auto role : roles) {
        values.push_back(static_cast<uint32_t>(role));
    }
    return values;
}

inline void gfx_mpsrt_set_stage_manifest_attrs(mlir::Operation* module,
                                               const std::string& prefix,
                                               const GfxKernelStageManifest& manifest) {
    if (!module || !manifest.valid) {
        return;
    }

    mlir::Builder builder(module->getContext());
    module->setAttr(prefix + ".stage_family",
                    builder.getStringAttr(gfx_kernel_stage_family_name(manifest.stage_family)));
    module->setAttr(prefix + ".backend_domain",
                    builder.getStringAttr(gfx_kernel_backend_domain_name(manifest.backend_domain)));
    module->setAttr(prefix + ".execution_kind",
                    builder.getStringAttr(gfx_kernel_execution_kind_name(manifest.execution_kind)));
    module->setAttr(prefix + ".storage",
                    builder.getStringAttr(gfx_kernel_storage_kind_name(manifest.storage)));
    if (!manifest.specialization_key.empty()) {
        module->setAttr(prefix + ".specialization_key",
                        builder.getStringAttr(manifest.specialization_key));
    }
    if (!manifest.semantic_input_roles.empty()) {
        module->setAttr(prefix + ".semantic_input_roles",
                        gfx_mpsrt_u32_vector_attr(
                            builder,
                            gfx_kernel_buffer_roles_to_attr_values(manifest.semantic_input_roles)));
    }
    if (!manifest.semantic_output_roles.empty()) {
        module->setAttr(prefix + ".semantic_output_roles",
                        gfx_mpsrt_u32_vector_attr(
                            builder,
                            gfx_kernel_buffer_roles_to_attr_values(manifest.semantic_output_roles)));
    }

    if (!manifest.custom_kernel.valid) {
        return;
    }

    module->setAttr(prefix + ".kernel.family",
                    builder.getStringAttr(manifest.custom_kernel.kernel_family));
    module->setAttr(prefix + ".kernel.family_id",
                    builder.getI32IntegerAttr(static_cast<int32_t>(manifest.custom_kernel.kernel_family_id)));
    module->setAttr(prefix + ".kernel.entry_point",
                    builder.getStringAttr(manifest.custom_kernel.entry_point));
    if (manifest.custom_kernel.dispatch_policy.valid) {
        const auto& dispatch_policy = manifest.custom_kernel.dispatch_policy;
        module->setAttr(prefix + ".kernel.dispatch_policy.grid",
                        builder.getStringAttr(gfx_kernel_dispatch_grid_name(dispatch_policy.grid)));
        module->setAttr(prefix + ".kernel.dispatch_policy.threads_per_threadgroup",
                        builder.getI32IntegerAttr(
                            static_cast<int32_t>(dispatch_policy.threads_per_threadgroup)));
        module->setAttr(prefix + ".kernel.dispatch_policy.precompiled_binary_required",
                        builder.getBoolAttr(dispatch_policy.precompiled_binary_required));
    }

    const auto& abi = manifest.custom_kernel.external_buffer_abi;
    if (!abi.valid) {
        return;
    }
    module->setAttr(prefix + ".kernel.external_buffer_abi.valid", builder.getBoolAttr(true));
    module->setAttr(prefix + ".kernel.external_buffer_abi.leading_input_count",
                    builder.getI32IntegerAttr(static_cast<int32_t>(abi.leading_input_count)));
    module->setAttr(prefix + ".kernel.external_buffer_abi.leading_output_count",
                    builder.getI32IntegerAttr(static_cast<int32_t>(abi.leading_output_count)));
    if (!abi.roles.empty()) {
        module->setAttr(prefix + ".kernel.external_buffer_abi.roles",
                        gfx_mpsrt_u32_vector_attr(builder, gfx_kernel_buffer_roles_to_attr_values(abi.roles)));
    }
}

inline bool gfx_mpsrt_read_stage_manifest_attrs(mlir::Operation* module,
                                                const std::string& prefix,
                                                GfxKernelStageManifest& manifest) {
    manifest = {};
    if (!module) {
        return false;
    }

    std::string stage_family;
    std::string backend_domain;
    std::string execution_kind;
    std::string storage;
    if (!gfx_mpsrt_read_string_attr(module, prefix + ".stage_family", stage_family) ||
        !gfx_mpsrt_read_string_attr(module, prefix + ".backend_domain", backend_domain) ||
        !gfx_mpsrt_read_string_attr(module, prefix + ".execution_kind", execution_kind) ||
        !gfx_mpsrt_read_string_attr(module, prefix + ".storage", storage)) {
        return false;
    }

    manifest.stage_family = gfx_kernel_stage_family_from_name(stage_family);
    manifest.backend_domain = gfx_kernel_backend_domain_from_name(backend_domain);
    manifest.execution_kind = gfx_kernel_execution_kind_from_name(execution_kind);
    manifest.storage = gfx_kernel_storage_kind_from_name(storage);
    (void)gfx_mpsrt_read_string_attr(module, prefix + ".specialization_key", manifest.specialization_key);
    const auto input_role_values = gfx_mpsrt_read_u32_vector_attr(module,
                                                                  prefix + ".semantic_input_roles");
    manifest.semantic_input_roles.reserve(input_role_values.size());
    for (const auto role_value : input_role_values) {
        manifest.semantic_input_roles.push_back(static_cast<GfxKernelBufferRole>(role_value));
    }
    const auto output_role_values = gfx_mpsrt_read_u32_vector_attr(module,
                                                                   prefix + ".semantic_output_roles");
    manifest.semantic_output_roles.reserve(output_role_values.size());
    for (const auto role_value : output_role_values) {
        manifest.semantic_output_roles.push_back(static_cast<GfxKernelBufferRole>(role_value));
    }
    manifest.valid = manifest.stage_family != GfxKernelStageFamily::Unknown &&
                     manifest.backend_domain != GfxKernelBackendDomain::Unknown &&
                     manifest.execution_kind != GfxKernelExecutionKind::Unknown &&
                     manifest.storage != GfxKernelStorageKind::Unknown;
    if (!manifest.valid) {
        return false;
    }

    if (manifest.execution_kind != GfxKernelExecutionKind::CustomKernel) {
        return true;
    }

    auto& custom = manifest.custom_kernel;
    std::string kernel_family;
    std::string entry_point;
    if (!gfx_mpsrt_read_string_attr(module, prefix + ".kernel.family", kernel_family) ||
        !gfx_mpsrt_read_string_attr(module, prefix + ".kernel.entry_point", entry_point)) {
        return true;
    }
    custom.valid = true;
    custom.kernel_family = kernel_family;
    custom.entry_point = entry_point;
    (void)gfx_mpsrt_read_i32_attr(module, prefix + ".kernel.family_id", custom.kernel_family_id);
    std::string dispatch_grid;
    uint32_t dispatch_threads = 0;
    bool dispatch_precompiled_required = false;
    const bool has_dispatch_grid =
        gfx_mpsrt_read_string_attr(module, prefix + ".kernel.dispatch_policy.grid", dispatch_grid);
    const bool has_dispatch_threads =
        gfx_mpsrt_read_i32_attr(module,
                                prefix + ".kernel.dispatch_policy.threads_per_threadgroup",
                                dispatch_threads);
    const bool has_dispatch_precompile =
        gfx_mpsrt_read_bool_attr(module,
                                 prefix + ".kernel.dispatch_policy.precompiled_binary_required",
                                 dispatch_precompiled_required);
    if (has_dispatch_grid || has_dispatch_threads || has_dispatch_precompile) {
        custom.dispatch_policy =
            make_gfx_kernel_dispatch_policy(gfx_kernel_dispatch_grid_from_name(dispatch_grid),
                                            dispatch_threads,
                                            dispatch_precompiled_required);
    }

    bool abi_valid = false;
    (void)gfx_mpsrt_read_bool_attr(module, prefix + ".kernel.external_buffer_abi.valid", abi_valid);
    if (abi_valid) {
        auto& abi = custom.external_buffer_abi;
        abi.valid = true;
        (void)gfx_mpsrt_read_i32_attr(module,
                                      prefix + ".kernel.external_buffer_abi.leading_input_count",
                                      abi.leading_input_count);
        (void)gfx_mpsrt_read_i32_attr(module,
                                      prefix + ".kernel.external_buffer_abi.leading_output_count",
                                      abi.leading_output_count);
        const auto role_values = gfx_mpsrt_read_u32_vector_attr(module,
                                                                prefix + ".kernel.external_buffer_abi.roles");
        abi.roles.reserve(role_values.size());
        for (const auto role_value : role_values) {
            abi.roles.push_back(static_cast<GfxKernelBufferRole>(role_value));
        }
    }
    return true;
}

inline void gfx_mpsrt_set_stage_manifest_attrs(mlir::Operation* module,
                                               const GfxKernelStageManifest& manifest) {
    gfx_mpsrt_set_stage_manifest_attrs(module, "gfx.stage_manifest", manifest);
}

inline bool gfx_mpsrt_read_stage_manifest_attrs(mlir::ModuleOp module,
                                                GfxKernelStageManifest& manifest) {
    return gfx_mpsrt_read_stage_manifest_attrs(module, "gfx.stage_manifest", manifest);
}

inline GfxStageBackendDomain gfx_mpsrt_stage_domain_from_kernel_domain(GfxKernelBackendDomain domain) {
    switch (domain) {
        case GfxKernelBackendDomain::AppleMps:
            return GfxStageBackendDomain::AppleMps;
        case GfxKernelBackendDomain::AppleMsl:
            return GfxStageBackendDomain::AppleMsl;
        case GfxKernelBackendDomain::Spirv:
            return GfxStageBackendDomain::Spirv;
        case GfxKernelBackendDomain::Unknown:
        default:
            return GfxStageBackendDomain::Unknown;
    }
}

inline void gfx_mpsrt_apply_stage_manifest_to_stage_desc(GfxMpsrtStageDesc& stage) {
    const auto& manifest = stage.stage_manifest;
    if (!manifest.valid) {
        return;
    }

    const auto manifest_kind = gfx_mpsrt_stage_kind_from_manifest(manifest);
    if (manifest_kind != GfxMpsrtStageKind::Unknown) {
        stage.kind = manifest_kind;
        stage.builder_symbol = gfx_mpsrt_builder_symbol(manifest_kind);
    }

    const auto manifest_domain = gfx_mpsrt_stage_domain_from_kernel_domain(manifest.backend_domain);
    if (manifest_domain != GfxStageBackendDomain::Unknown) {
        stage.domain = manifest_domain;
    }
    stage.uses_vendor_primitive = manifest.execution_kind == GfxKernelExecutionKind::VendorPrimitive;
    stage.uses_custom_kernel = manifest.execution_kind == GfxKernelExecutionKind::CustomKernel;
    if (!manifest.specialization_key.empty()) {
        stage.specialization_key = manifest.specialization_key;
    }

    const auto& custom = manifest.custom_kernel;
    if (!custom.valid) {
        return;
    }
    const auto dispatch = gfx_mpsrt_custom_dispatch_spec_from_kernel_manifest(custom);
    if (dispatch.valid) {
        stage.dispatch_kernel_family = dispatch.kernel_family;
        stage.dispatch_entry_point = dispatch.entry_point;
        if (!dispatch.entry_point.empty()) {
            stage.kernel_name = dispatch.entry_point;
        }
        stage.dispatch_kernel_family_id = dispatch.kernel_family_id;
        stage.dispatch_threads_per_threadgroup = dispatch.threads_per_threadgroup;
        stage.dispatch_precompiled_kernel_required = dispatch.precompiled_binary_required;
        stage.dispatch_flags = dispatch.flags;
    }
}

inline void gfx_mpsrt_set_tensor_desc_attrs(mlir::Operation* module,
                                            std::string prefix,
                                            const GfxMpsrtTensorDesc& desc) {
    mlir::Builder builder(module->getContext());
    module->setAttr(prefix + ".dtype",
                    builder.getStringAttr(gfx_mpsrt_dtype_name(desc.dtype)));
    module->setAttr(prefix + ".storage",
                    builder.getStringAttr(gfx_mpsrt_storage_name(desc.storage)));
    module->setAttr(prefix + ".layout",
                    builder.getStringAttr(gfx_mpsrt_layout_name(desc.layout)));
    module->setAttr(prefix + ".rank", builder.getI32IntegerAttr(static_cast<int32_t>(desc.rank)));
    module->setAttr(prefix + ".dims", gfx_mpsrt_i64_array_attr(builder, desc.dims.data(), desc.rank));
    module->setAttr(prefix + ".strides", gfx_mpsrt_i64_array_attr(builder, desc.strides.data(), desc.rank));
    module->setAttr(prefix + ".flags", builder.getI32IntegerAttr(static_cast<int32_t>(desc.flags)));
    module->setAttr(prefix + ".byte_length",
                    builder.getI64IntegerAttr(static_cast<int64_t>(desc.byte_length)));
    if (desc.storage == GfxMpsrtStorage::Image) {
        module->setAttr(prefix + ".image_width",
                        builder.getI32IntegerAttr(static_cast<int32_t>(desc.image_width)));
        module->setAttr(prefix + ".image_height",
                        builder.getI32IntegerAttr(static_cast<int32_t>(desc.image_height)));
        module->setAttr(prefix + ".image_feature_channels",
                        builder.getI32IntegerAttr(static_cast<int32_t>(desc.image_feature_channels)));
        module->setAttr(prefix + ".image_batch",
                        builder.getI32IntegerAttr(static_cast<int32_t>(desc.image_batch)));
    } else if (desc.storage == GfxMpsrtStorage::Matrix) {
        module->setAttr(prefix + ".matrix_rows",
                        builder.getI32IntegerAttr(static_cast<int32_t>(desc.matrix_rows)));
        module->setAttr(prefix + ".matrix_columns",
                        builder.getI32IntegerAttr(static_cast<int32_t>(desc.matrix_columns)));
        module->setAttr(prefix + ".matrix_row_bytes",
                        builder.getI32IntegerAttr(static_cast<int32_t>(desc.matrix_row_bytes)));
        module->setAttr(prefix + ".matrix_count",
                        builder.getI32IntegerAttr(static_cast<int32_t>(desc.matrix_count)));
    }
}

inline bool gfx_mpsrt_read_tensor_desc_attrs(mlir::Operation* module,
                                             const std::string& prefix,
                                             GfxMpsrtTensorDesc& desc) {
    std::string dtype;
    std::string storage;
    std::string layout;
    if (!gfx_mpsrt_read_string_attr(module, prefix + ".dtype", dtype) ||
        !gfx_mpsrt_read_string_attr(module, prefix + ".storage", storage) ||
        !gfx_mpsrt_read_string_attr(module, prefix + ".layout", layout) ||
        !gfx_mpsrt_read_i32_attr(module, prefix + ".rank", desc.rank)) {
        return false;
    }

    desc.dtype = gfx_mpsrt_dtype_from_name(dtype);
    desc.storage = gfx_mpsrt_storage_from_name(storage);
    desc.layout = gfx_mpsrt_layout_from_name(layout);
    desc.rank = std::min<uint32_t>(desc.rank, static_cast<uint32_t>(desc.dims.size()));
    gfx_mpsrt_read_u64_array_attr(module, prefix + ".dims", desc.dims.data(), desc.rank);
    gfx_mpsrt_read_i64_array_attr(module, prefix + ".strides", desc.strides.data(), desc.rank);
    (void)gfx_mpsrt_read_i32_attr(module, prefix + ".flags", desc.flags);
    (void)gfx_mpsrt_read_i64_attr(module, prefix + ".byte_length", desc.byte_length);
    (void)gfx_mpsrt_read_i64_attr(module, prefix + ".byte_offset", desc.byte_offset);
    (void)gfx_mpsrt_read_i32_attr(module, prefix + ".image_width", desc.image_width);
    (void)gfx_mpsrt_read_i32_attr(module, prefix + ".image_height", desc.image_height);
    (void)gfx_mpsrt_read_i32_attr(module, prefix + ".image_feature_channels", desc.image_feature_channels);
    (void)gfx_mpsrt_read_i32_attr(module, prefix + ".image_batch", desc.image_batch);
    (void)gfx_mpsrt_read_i32_attr(module, prefix + ".matrix_rows", desc.matrix_rows);
    (void)gfx_mpsrt_read_i32_attr(module, prefix + ".matrix_columns", desc.matrix_columns);
    (void)gfx_mpsrt_read_i32_attr(module, prefix + ".matrix_row_bytes", desc.matrix_row_bytes);
    (void)gfx_mpsrt_read_i32_attr(module, prefix + ".matrix_count", desc.matrix_count);
    (void)gfx_mpsrt_read_i32_attr(module, prefix + ".alias_of", desc.alias_of);
    return true;
}

inline void gfx_mpsrt_set_storage_bridge_attrs(mlir::Operation* module,
                                               const std::string& prefix,
                                               const GfxMpsrtStorageBridgeDesc& bridge) {
    mlir::Builder builder(module->getContext());
    module->setAttr(prefix + ".value",
                    builder.getI32IntegerAttr(static_cast<int32_t>(bridge.value)));
    module->setAttr(prefix + ".direction",
                    builder.getStringAttr(gfx_mpsrt_storage_bridge_direction_name(bridge.direction)));
    module->setAttr(prefix + ".source_storage",
                    builder.getStringAttr(gfx_mpsrt_storage_name(bridge.source_storage)));
    module->setAttr(prefix + ".target_storage",
                    builder.getStringAttr(gfx_mpsrt_storage_name(bridge.target_storage)));
    gfx_mpsrt_set_tensor_desc_attrs(module, prefix + ".tensor", gfx_mpsrt_from_abi_desc(bridge.tensor));
}

inline bool gfx_mpsrt_read_storage_bridge_attrs(mlir::Operation* module,
                                                const std::string& prefix,
                                                GfxMpsrtStorageBridgeDesc& bridge) {
    uint32_t value = 0;
    std::string direction_name;
    std::string source_storage_name;
    std::string target_storage_name;
    GfxMpsrtTensorDesc tensor{};
    if (!gfx_mpsrt_read_i32_attr(module, prefix + ".value", value) ||
        !gfx_mpsrt_read_string_attr(module, prefix + ".direction", direction_name) ||
        !gfx_mpsrt_read_string_attr(module, prefix + ".source_storage", source_storage_name) ||
        !gfx_mpsrt_read_string_attr(module, prefix + ".target_storage", target_storage_name) ||
        !gfx_mpsrt_read_tensor_desc_attrs(module, prefix + ".tensor", tensor)) {
        return false;
    }

    const auto direction = gfx_mpsrt_storage_bridge_direction_from_name(direction_name);
    const auto source_storage = gfx_mpsrt_storage_from_name(source_storage_name);
    const auto target_storage = gfx_mpsrt_storage_from_name(target_storage_name);
    GfxMpsrtStorageBridgeDesc normalized{};
    if (!gfx_mpsrt_make_storage_bridge_desc(value,
                                            gfx_mpsrt_to_abi_desc(tensor),
                                            direction,
                                            normalized)) {
        return false;
    }
    if (normalized.source_storage != source_storage ||
        normalized.target_storage != target_storage) {
        return false;
    }

    bridge = normalized;
    return true;
}

inline void gfx_mpsrt_set_storage_bridges_attrs(mlir::Operation* module,
                                                const std::string& prefix,
                                                const std::vector<GfxMpsrtStorageBridgeDesc>& bridges) {
    mlir::Builder builder(module->getContext());
    module->setAttr(prefix + ".storage_bridge_count",
                    builder.getI32IntegerAttr(static_cast<int32_t>(bridges.size())));
    for (size_t i = 0; i < bridges.size(); ++i) {
        gfx_mpsrt_set_storage_bridge_attrs(module,
                                           prefix + ".storage_bridge" + std::to_string(i),
                                           bridges[i]);
    }
}

inline bool gfx_mpsrt_read_storage_bridges_attrs(mlir::Operation* module,
                                                 const std::string& prefix,
                                                 std::vector<GfxMpsrtStorageBridgeDesc>& bridges) {
    bridges.clear();
    uint32_t bridge_count = 0;
    if (!gfx_mpsrt_read_i32_attr(module, prefix + ".storage_bridge_count", bridge_count)) {
        return false;
    }
    bridges.reserve(bridge_count);
    for (uint32_t i = 0; i < bridge_count; ++i) {
        GfxMpsrtStorageBridgeDesc bridge{};
        if (!gfx_mpsrt_read_storage_bridge_attrs(module,
                                                 prefix + ".storage_bridge" + std::to_string(i),
                                                 bridge)) {
            bridges.clear();
            return false;
        }
        bridges.push_back(bridge);
    }
    return true;
}

inline void gfx_mpsrt_set_storage_bridges_attrs(mlir::Operation* module,
                                                const std::vector<GfxMpsrtStorageBridgeDesc>& bridges) {
    gfx_mpsrt_set_storage_bridges_attrs(module, "gfx.mpsrt", bridges);
}

inline bool gfx_mpsrt_read_storage_bridges_attrs(mlir::Operation* module,
                                                 std::vector<GfxMpsrtStorageBridgeDesc>& bridges) {
    return gfx_mpsrt_read_storage_bridges_attrs(module, "gfx.mpsrt", bridges);
}

inline void gfx_mpsrt_set_gemm_desc_attrs(mlir::Operation* module,
                                          const std::string& prefix,
                                          const GfxMpsrtGemmAbiDesc& desc) {
    mlir::Builder builder(module->getContext());
    module->setAttr(prefix + ".transpose_lhs",
                    builder.getI32IntegerAttr(static_cast<int32_t>(desc.transpose_lhs)));
    module->setAttr(prefix + ".transpose_rhs",
                    builder.getI32IntegerAttr(static_cast<int32_t>(desc.transpose_rhs)));
    module->setAttr(prefix + ".accumulate_fp32",
                    builder.getI32IntegerAttr(static_cast<int32_t>(desc.accumulate_fp32)));
    module->setAttr(prefix + ".alpha", builder.getF32FloatAttr(desc.alpha));
    module->setAttr(prefix + ".beta", builder.getF32FloatAttr(desc.beta));
}

inline bool gfx_mpsrt_read_gemm_desc_attrs(mlir::Operation* module,
                                           const std::string& prefix,
                                           GfxMpsrtGemmAbiDesc& desc) {
    bool read_any = false;
    read_any |= gfx_mpsrt_read_i32_attr(module, prefix + ".transpose_lhs", desc.transpose_lhs);
    read_any |= gfx_mpsrt_read_i32_attr(module, prefix + ".transpose_rhs", desc.transpose_rhs);
    read_any |= gfx_mpsrt_read_i32_attr(module, prefix + ".accumulate_fp32", desc.accumulate_fp32);
    if (auto attr = module->getAttrOfType<mlir::FloatAttr>(prefix + ".alpha")) {
        desc.alpha = static_cast<float>(attr.getValueAsDouble());
        read_any = true;
    }
    if (auto attr = module->getAttrOfType<mlir::FloatAttr>(prefix + ".beta")) {
        desc.beta = static_cast<float>(attr.getValueAsDouble());
        read_any = true;
    }
    return read_any;
}

inline void gfx_mpsrt_set_pool2d_desc_attrs(mlir::Operation* module,
                                            const std::string& prefix,
                                            const GfxMpsrtPool2DAbiDesc& desc) {
    mlir::Builder builder(module->getContext());
    const uint64_t kernel[] = {desc.kernel[0], desc.kernel[1]};
    const uint64_t strides[] = {desc.strides[0], desc.strides[1]};
    const uint64_t dilations[] = {desc.dilations[0], desc.dilations[1]};
    const uint64_t pads[] = {desc.pads[0], desc.pads[1], desc.pads[2], desc.pads[3]};
    module->setAttr(prefix + ".is_avg",
                    builder.getI32IntegerAttr(static_cast<int32_t>(desc.is_avg)));
    module->setAttr(prefix + ".kernel",
                    gfx_mpsrt_i64_array_attr(builder, kernel, 2));
    module->setAttr(prefix + ".strides",
                    gfx_mpsrt_i64_array_attr(builder, strides, 2));
    module->setAttr(prefix + ".dilations",
                    gfx_mpsrt_i64_array_attr(builder, dilations, 2));
    module->setAttr(prefix + ".pads",
                    gfx_mpsrt_i64_array_attr(builder, pads, 4));
    module->setAttr(prefix + ".exclude_pad",
                    builder.getI32IntegerAttr(static_cast<int32_t>(desc.exclude_pad)));
}

inline bool gfx_mpsrt_read_pool2d_desc_attrs(mlir::Operation* module,
                                             const std::string& prefix,
                                             GfxMpsrtPool2DAbiDesc& desc) {
    bool read_any = false;
    uint64_t kernel[] = {desc.kernel[0], desc.kernel[1]};
    uint64_t strides[] = {desc.strides[0], desc.strides[1]};
    uint64_t dilations[] = {desc.dilations[0], desc.dilations[1]};
    uint64_t pads[] = {desc.pads[0], desc.pads[1], desc.pads[2], desc.pads[3]};
    read_any |= gfx_mpsrt_read_i32_attr(module, prefix + ".is_avg", desc.is_avg);
    gfx_mpsrt_read_u64_array_attr(module, prefix + ".kernel", kernel, 2);
    read_any = read_any || module->hasAttr(prefix + ".kernel");
    gfx_mpsrt_read_u64_array_attr(module, prefix + ".strides", strides, 2);
    read_any = read_any || module->hasAttr(prefix + ".strides");
    gfx_mpsrt_read_u64_array_attr(module, prefix + ".dilations", dilations, 2);
    read_any = read_any || module->hasAttr(prefix + ".dilations");
    gfx_mpsrt_read_u64_array_attr(module, prefix + ".pads", pads, 4);
    read_any = read_any || module->hasAttr(prefix + ".pads");
    read_any |= gfx_mpsrt_read_i32_attr(module, prefix + ".exclude_pad", desc.exclude_pad);
    desc.kernel[0] = static_cast<uint32_t>(kernel[0]);
    desc.kernel[1] = static_cast<uint32_t>(kernel[1]);
    desc.strides[0] = static_cast<uint32_t>(strides[0]);
    desc.strides[1] = static_cast<uint32_t>(strides[1]);
    desc.dilations[0] = static_cast<uint32_t>(dilations[0]);
    desc.dilations[1] = static_cast<uint32_t>(dilations[1]);
    for (size_t i = 0; i < 4; ++i) {
        desc.pads[i] = static_cast<uint32_t>(pads[i]);
    }
    return read_any;
}

inline void gfx_mpsrt_set_softmax_desc_attrs(mlir::Operation* module,
                                             const std::string& prefix,
                                             const GfxMpsrtSoftmaxAbiDesc& desc) {
    mlir::Builder builder(module->getContext());
    module->setAttr(prefix + ".axis",
                    builder.getI32IntegerAttr(static_cast<int32_t>(desc.axis)));
    module->setAttr(prefix + ".log_softmax",
                    builder.getI32IntegerAttr(static_cast<int32_t>(desc.log_softmax)));
}

inline bool gfx_mpsrt_read_softmax_desc_attrs(mlir::Operation* module,
                                              const std::string& prefix,
                                              GfxMpsrtSoftmaxAbiDesc& desc) {
    bool read_any = false;
    read_any |= gfx_mpsrt_read_i32_attr(module, prefix + ".axis", desc.axis);
    read_any |= gfx_mpsrt_read_i32_attr(module, prefix + ".log_softmax", desc.log_softmax);
    return read_any;
}

inline void gfx_mpsrt_set_topk_desc_attrs(mlir::Operation* module,
                                          const std::string& prefix,
                                          const GfxMpsrtTopKAbiDesc& desc) {
    mlir::Builder builder(module->getContext());
    module->setAttr(prefix + ".axis",
                    builder.getI32IntegerAttr(static_cast<int32_t>(desc.axis)));
    module->setAttr(prefix + ".k",
                    builder.getI32IntegerAttr(static_cast<int32_t>(desc.k)));
    module->setAttr(prefix + ".mode_max",
                    builder.getI32IntegerAttr(static_cast<int32_t>(desc.mode_max)));
    module->setAttr(prefix + ".sort_type",
                    builder.getI32IntegerAttr(static_cast<int32_t>(desc.sort_type)));
}

inline bool gfx_mpsrt_read_topk_desc_attrs(mlir::Operation* module,
                                           const std::string& prefix,
                                           GfxMpsrtTopKAbiDesc& desc) {
    bool read_any = false;
    read_any |= gfx_mpsrt_read_i32_attr(module, prefix + ".axis", desc.axis);
    read_any |= gfx_mpsrt_read_i32_attr(module, prefix + ".k", desc.k);
    read_any |= gfx_mpsrt_read_i32_attr(module, prefix + ".mode_max", desc.mode_max);
    read_any |= gfx_mpsrt_read_i32_attr(module, prefix + ".sort_type", desc.sort_type);
    return read_any;
}

inline void gfx_mpsrt_set_conv2d_desc_attrs(mlir::Operation* module,
                                            const std::string& prefix,
                                            const GfxMpsrtConv2DAbiDesc& desc) {
    mlir::Builder builder(module->getContext());
    const uint64_t strides[] = {desc.strides[0], desc.strides[1]};
    const uint64_t dilations[] = {desc.dilations[0], desc.dilations[1]};
    const uint64_t pads[] = {desc.pads[0], desc.pads[1], desc.pads[2], desc.pads[3]};
    module->setAttr(prefix + ".groups",
                    builder.getI32IntegerAttr(static_cast<int32_t>(desc.groups)));
    module->setAttr(prefix + ".strides",
                    gfx_mpsrt_i64_array_attr(builder, strides, 2));
    module->setAttr(prefix + ".dilations",
                    gfx_mpsrt_i64_array_attr(builder, dilations, 2));
    module->setAttr(prefix + ".pads",
                    gfx_mpsrt_i64_array_attr(builder, pads, 4));
    module->setAttr(prefix + ".fused_activation",
                    builder.getI32IntegerAttr(static_cast<int32_t>(desc.fused_activation)));
    module->setAttr(prefix + ".accumulate_fp32",
                    builder.getI32IntegerAttr(static_cast<int32_t>(desc.accumulate_fp32)));
}

inline bool gfx_mpsrt_read_conv2d_desc_attrs(mlir::Operation* module,
                                             const std::string& prefix,
                                             GfxMpsrtConv2DAbiDesc& desc) {
    bool read_any = false;
    uint64_t strides[] = {desc.strides[0], desc.strides[1]};
    uint64_t dilations[] = {desc.dilations[0], desc.dilations[1]};
    uint64_t pads[] = {desc.pads[0], desc.pads[1], desc.pads[2], desc.pads[3]};
    read_any |= gfx_mpsrt_read_i32_attr(module, prefix + ".groups", desc.groups);
    gfx_mpsrt_read_u64_array_attr(module, prefix + ".strides", strides, 2);
    read_any = read_any || module->hasAttr(prefix + ".strides");
    gfx_mpsrt_read_u64_array_attr(module, prefix + ".dilations", dilations, 2);
    read_any = read_any || module->hasAttr(prefix + ".dilations");
    gfx_mpsrt_read_u64_array_attr(module, prefix + ".pads", pads, 4);
    read_any = read_any || module->hasAttr(prefix + ".pads");
    read_any |= gfx_mpsrt_read_i32_attr(module, prefix + ".fused_activation", desc.fused_activation);
    read_any |= gfx_mpsrt_read_i32_attr(module, prefix + ".accumulate_fp32", desc.accumulate_fp32);
    desc.strides[0] = static_cast<uint32_t>(strides[0]);
    desc.strides[1] = static_cast<uint32_t>(strides[1]);
    desc.dilations[0] = static_cast<uint32_t>(dilations[0]);
    desc.dilations[1] = static_cast<uint32_t>(dilations[1]);
    for (size_t i = 0; i < 4; ++i) {
        desc.pads[i] = static_cast<uint32_t>(pads[i]);
    }
    return read_any;
}

inline void gfx_mpsrt_set_conv2d_desc_attrs(mlir::Operation* module,
                                            const GfxMpsrtConv2DAbiDesc& desc) {
    gfx_mpsrt_set_conv2d_desc_attrs(module, "gfx.mpsrt.conv2d", desc);
}

inline bool gfx_mpsrt_read_conv2d_desc_attrs(mlir::Operation* module,
                                             GfxMpsrtConv2DAbiDesc& desc) {
    return gfx_mpsrt_read_conv2d_desc_attrs(module, "gfx.mpsrt.conv2d", desc);
}

inline void gfx_mpsrt_set_gemm_desc_attrs(mlir::Operation* module,
                                          const GfxMpsrtGemmAbiDesc& desc) {
    gfx_mpsrt_set_gemm_desc_attrs(module, "gfx.mpsrt.gemm", desc);
}

inline bool gfx_mpsrt_read_gemm_desc_attrs(mlir::Operation* module,
                                           GfxMpsrtGemmAbiDesc& desc) {
    return gfx_mpsrt_read_gemm_desc_attrs(module, "gfx.mpsrt.gemm", desc);
}

inline void gfx_mpsrt_set_pool2d_desc_attrs(mlir::Operation* module,
                                            const GfxMpsrtPool2DAbiDesc& desc) {
    gfx_mpsrt_set_pool2d_desc_attrs(module, "gfx.mpsrt.pool2d", desc);
}

inline bool gfx_mpsrt_read_pool2d_desc_attrs(mlir::Operation* module,
                                             GfxMpsrtPool2DAbiDesc& desc) {
    return gfx_mpsrt_read_pool2d_desc_attrs(module, "gfx.mpsrt.pool2d", desc);
}

inline void gfx_mpsrt_set_softmax_desc_attrs(mlir::Operation* module,
                                             const GfxMpsrtSoftmaxAbiDesc& desc) {
    gfx_mpsrt_set_softmax_desc_attrs(module, "gfx.mpsrt.softmax", desc);
}

inline bool gfx_mpsrt_read_softmax_desc_attrs(mlir::Operation* module,
                                              GfxMpsrtSoftmaxAbiDesc& desc) {
    return gfx_mpsrt_read_softmax_desc_attrs(module, "gfx.mpsrt.softmax", desc);
}

inline void gfx_mpsrt_set_topk_desc_attrs(mlir::Operation* module,
                                          const GfxMpsrtTopKAbiDesc& desc) {
    gfx_mpsrt_set_topk_desc_attrs(module, "gfx.mpsrt.topk", desc);
}

inline bool gfx_mpsrt_read_topk_desc_attrs(mlir::Operation* module,
                                           GfxMpsrtTopKAbiDesc& desc) {
    return gfx_mpsrt_read_topk_desc_attrs(module, "gfx.mpsrt.topk", desc);
}

inline std::vector<GfxKernelBufferRole> gfx_mpsrt_normalize_semantic_input_roles(
    size_t input_count,
    const std::vector<GfxKernelBufferRole>& roles);

inline bool gfx_mpsrt_collect_entry_tensors(mlir::ModuleOp module,
                                            const GfxStageOptimizationPlan& plan,
                                            std::vector<GfxMpsrtTensorDesc>& inputs,
                                            std::vector<GfxMpsrtTensorDesc>& outputs,
                                            const std::vector<GfxKernelBufferRole>& semantic_arg_roles = {}) {
    inputs.clear();
    outputs.clear();
    auto func = gfx_mpsrt_entry_func(module);
    if (!func) {
        return static_cast<bool>(module);
    }

    const auto fn_type = func.getFunctionType();
    const auto storage = plan.placement.storage;
    inputs.reserve(fn_type.getNumInputs());
    outputs.reserve(fn_type.getNumResults());
    if (fn_type.getNumResults() == 0 && !semantic_arg_roles.empty()) {
        const auto roles = gfx_mpsrt_normalize_semantic_input_roles(fn_type.getNumInputs(),
                                                                    semantic_arg_roles);
        for (unsigned i = 0; i < fn_type.getNumInputs(); ++i) {
            const auto type = fn_type.getInput(i);
            const auto role = roles[i];
            if (role == GfxKernelBufferRole::RuntimeParams) {
                continue;
            }
            auto desc = gfx_mpsrt_make_tensor_desc(gfx_mpsrt_shape_from_mlir_type(type),
                                                   gfx_mpsrt_element_from_mlir_type(type),
                                                   storage,
                                                   role == GfxKernelBufferRole::TensorOutput
                                                       ? GfxMpsrtTensorFlagTransient
                                                       : GfxMpsrtTensorFlagExternalIo);
            if (role == GfxKernelBufferRole::TensorOutput) {
                outputs.push_back(std::move(desc));
            } else {
                inputs.push_back(std::move(desc));
            }
        }
        return true;
    }
    for (unsigned i = 0; i < fn_type.getNumInputs(); ++i) {
        const auto type = fn_type.getInput(i);
        inputs.push_back(gfx_mpsrt_make_tensor_desc(gfx_mpsrt_shape_from_mlir_type(type),
                                                    gfx_mpsrt_element_from_mlir_type(type),
                                                    storage,
                                                    GfxMpsrtTensorFlagExternalIo));
    }
    for (unsigned i = 0; i < fn_type.getNumResults(); ++i) {
        const auto type = fn_type.getResult(i);
        outputs.push_back(gfx_mpsrt_make_tensor_desc(gfx_mpsrt_shape_from_mlir_type(type),
                                                     gfx_mpsrt_element_from_mlir_type(type),
                                                     storage,
                                                     GfxMpsrtTensorFlagTransient));
    }
    return true;
}

inline std::vector<GfxKernelBufferRole> gfx_mpsrt_normalize_semantic_input_roles(
    size_t input_count,
    const std::vector<GfxKernelBufferRole>& roles) {
    std::vector<GfxKernelBufferRole> normalized(input_count, GfxKernelBufferRole::TensorInput);
    for (size_t i = 0; i < input_count && i < roles.size(); ++i) {
        if (roles[i] != GfxKernelBufferRole::Unknown) {
            normalized[i] = roles[i];
        }
    }
    return normalized;
}

inline std::vector<GfxKernelBufferRole> gfx_mpsrt_default_semantic_output_roles(size_t output_count) {
    return std::vector<GfxKernelBufferRole>(output_count, GfxKernelBufferRole::TensorOutput);
}

inline void gfx_mpsrt_reset_storage_specific_fields(GfxMpsrtTensorDesc& desc) {
    desc.image_width = 0;
    desc.image_height = 0;
    desc.image_feature_channels = 0;
    desc.image_batch = 0;
    desc.matrix_rows = 0;
    desc.matrix_columns = 0;
    desc.matrix_row_bytes = 0;
    desc.matrix_count = 0;
    desc.alias_of = 0;
}

inline void gfx_mpsrt_make_buffer_contract(GfxMpsrtTensorDesc& desc) {
    desc.storage = GfxMpsrtStorage::Buffer;
    desc.layout = GfxMpsrtLayout::Linear;
    gfx_mpsrt_reset_storage_specific_fields(desc);
}

inline void gfx_mpsrt_apply_semantic_input_roles(
    std::vector<GfxMpsrtTensorDesc>& inputs,
    const std::vector<GfxKernelBufferRole>& semantic_input_roles) {
    const auto roles = gfx_mpsrt_normalize_semantic_input_roles(inputs.size(),
                                                               semantic_input_roles);
    for (size_t input_index = 0; input_index < inputs.size(); ++input_index) {
        auto& desc = inputs[input_index];
        switch (roles[input_index]) {
            case GfxKernelBufferRole::ConstTensor:
                gfx_mpsrt_make_buffer_contract(desc);
                desc.flags = (desc.flags & GfxMpsrtTensorFlagDynamicShape) |
                             GfxMpsrtTensorFlagConst;
                break;
            case GfxKernelBufferRole::RuntimeParams:
                gfx_mpsrt_make_buffer_contract(desc);
                desc.flags = (desc.flags & GfxMpsrtTensorFlagDynamicShape) |
                             GfxMpsrtTensorFlagCpuVisible;
                break;
            case GfxKernelBufferRole::TensorInput:
            case GfxKernelBufferRole::TensorOutput:
            case GfxKernelBufferRole::Unknown:
            default:
                desc.flags &= ~GfxMpsrtTensorFlagConst;
                desc.flags |= GfxMpsrtTensorFlagExternalIo;
                break;
        }
    }
}

inline void gfx_mpsrt_set_stage_desc_attrs(mlir::Operation* module,
                                           const std::string& prefix,
                                           const GfxMpsrtStageDesc& stage,
                                           const std::string& stage_record_key) {
    mlir::Builder builder(module->getContext());
    module->setAttr(prefix + ".backend", builder.getStringAttr(gfx_stage_backend_domain_name(stage.domain)));
    module->setAttr(prefix + ".stage_kind", builder.getStringAttr(gfx_mpsrt_stage_kind_name(stage.kind)));
    module->setAttr(prefix + ".stage_record_key", builder.getStringAttr(stage_record_key));
    if (stage.input_storage != GfxMpsrtStorage::Unknown) {
        module->setAttr(prefix + ".input_storage", builder.getStringAttr(gfx_mpsrt_storage_name(stage.input_storage)));
    }
    if (stage.output_storage != GfxMpsrtStorage::Unknown) {
        module->setAttr(prefix + ".output_storage",
                        builder.getStringAttr(gfx_mpsrt_storage_name(stage.output_storage)));
    }
    if (stage.layout != GfxMpsrtLayout::Unknown) {
        module->setAttr(prefix + ".layout", builder.getStringAttr(gfx_mpsrt_layout_name(stage.layout)));
    }
    if (!stage.stage_type.empty()) {
        module->setAttr(prefix + ".stage_type", builder.getStringAttr(stage.stage_type));
    }
    if (!stage.kernel_name.empty()) {
        module->setAttr(prefix + ".kernel_name", builder.getStringAttr(stage.kernel_name));
    }
    if (!stage.builder_symbol.empty()) {
        module->setAttr(prefix + ".builder_symbol", builder.getStringAttr(stage.builder_symbol));
    }
    if (!stage.specialization_key.empty()) {
        module->setAttr(prefix + ".specialization_key", builder.getStringAttr(stage.specialization_key));
    }
    if (!stage.dispatch_kernel_family.empty()) {
        module->setAttr(prefix + ".dispatch_kernel_family",
                        builder.getStringAttr(stage.dispatch_kernel_family));
    }
    if (!stage.dispatch_entry_point.empty()) {
        module->setAttr(prefix + ".dispatch_entry_point",
                        builder.getStringAttr(stage.dispatch_entry_point));
    }
    if (stage.dispatch_kernel_family_id != 0) {
        module->setAttr(prefix + ".dispatch_kernel_family_id",
                        builder.getI32IntegerAttr(static_cast<int32_t>(stage.dispatch_kernel_family_id)));
    }
    if (stage.dispatch_flags != GfxMpsrtMslDispatchFlagNone) {
        module->setAttr(prefix + ".dispatch_flags",
                        builder.getI32IntegerAttr(static_cast<int32_t>(stage.dispatch_flags)));
    }
    if (stage.dispatch_threads_per_threadgroup != 0) {
        module->setAttr(prefix + ".dispatch_threads_per_threadgroup",
                        builder.getI32IntegerAttr(static_cast<int32_t>(stage.dispatch_threads_per_threadgroup)));
    }
    if (stage.dispatch_precompiled_kernel_required) {
        module->setAttr(prefix + ".dispatch_precompiled_kernel_required", builder.getBoolAttr(true));
    }
    module->setAttr(prefix + ".uses_vendor_primitive", builder.getBoolAttr(stage.uses_vendor_primitive));
    module->setAttr(prefix + ".uses_custom_kernel", builder.getBoolAttr(stage.uses_custom_kernel));
    if (stage.kind == GfxMpsrtStageKind::MPSConv2D ||
        stage.kind == GfxMpsrtStageKind::MPSGroupConv2D) {
        gfx_mpsrt_set_conv2d_desc_attrs(module, prefix + ".conv2d", stage.conv2d_desc);
    }
    if (stage.kind == GfxMpsrtStageKind::MPSGemm) {
        gfx_mpsrt_set_gemm_desc_attrs(module, prefix + ".gemm", stage.gemm_desc);
    }
    if (stage.kind == GfxMpsrtStageKind::MPSPool2D) {
        gfx_mpsrt_set_pool2d_desc_attrs(module, prefix + ".pool2d", stage.pool2d_desc);
    }
    if (stage.kind == GfxMpsrtStageKind::MPSSoftmax) {
        gfx_mpsrt_set_softmax_desc_attrs(module, prefix + ".softmax", stage.softmax_desc);
    }
    if (stage.kind == GfxMpsrtStageKind::MPSTopK) {
        gfx_mpsrt_set_topk_desc_attrs(module, prefix + ".topk", stage.topk_desc);
    }
}

inline bool gfx_mpsrt_read_stage_desc_attrs(mlir::Operation* module,
                                            const std::string& prefix,
                                            GfxMpsrtStageDesc& stage,
                                            std::string& stage_record_key) {
    stage = {};
    stage_record_key.clear();
    if (!module) {
        return false;
    }

    GfxKernelStageManifest canonical_manifest{};
    if (!gfx_mpsrt_read_stage_manifest_attrs(module, "gfx.stage_manifest", canonical_manifest)) {
        return false;
    }
    stage.stage_manifest = std::move(canonical_manifest);

    std::string input_storage;
    std::string output_storage;
    std::string layout;
    if (gfx_mpsrt_read_string_attr(module, prefix + ".input_storage", input_storage)) {
        stage.input_storage = gfx_mpsrt_storage_from_name(input_storage);
    }
    if (gfx_mpsrt_read_string_attr(module, prefix + ".output_storage", output_storage)) {
        stage.output_storage = gfx_mpsrt_storage_from_name(output_storage);
    }
    if (gfx_mpsrt_read_string_attr(module, prefix + ".layout", layout)) {
        stage.layout = gfx_mpsrt_layout_from_name(layout);
    }
    (void)gfx_mpsrt_read_string_attr(module, prefix + ".stage_type", stage.stage_type);
    (void)gfx_mpsrt_read_string_attr(module, prefix + ".kernel_name", stage.kernel_name);
    (void)gfx_mpsrt_read_string_attr(module, prefix + ".builder_symbol", stage.builder_symbol);
    (void)gfx_mpsrt_read_string_attr(module, prefix + ".specialization_key", stage.specialization_key);
    (void)gfx_mpsrt_read_string_attr(module, prefix + ".dispatch_kernel_family", stage.dispatch_kernel_family);
    (void)gfx_mpsrt_read_string_attr(module, prefix + ".dispatch_entry_point", stage.dispatch_entry_point);
    (void)gfx_mpsrt_read_i32_attr(module, prefix + ".dispatch_kernel_family_id", stage.dispatch_kernel_family_id);
    (void)gfx_mpsrt_read_i32_attr(module, prefix + ".dispatch_flags", stage.dispatch_flags);
    (void)gfx_mpsrt_read_i32_attr(module,
                                  prefix + ".dispatch_threads_per_threadgroup",
                                  stage.dispatch_threads_per_threadgroup);
    (void)gfx_mpsrt_read_bool_attr(module,
                                   prefix + ".dispatch_precompiled_kernel_required",
                                   stage.dispatch_precompiled_kernel_required);
    (void)gfx_mpsrt_read_bool_attr(module, prefix + ".uses_vendor_primitive", stage.uses_vendor_primitive);
    (void)gfx_mpsrt_read_bool_attr(module, prefix + ".uses_custom_kernel", stage.uses_custom_kernel);
    gfx_mpsrt_apply_stage_manifest_to_stage_desc(stage);
    if (stage.kind == GfxMpsrtStageKind::MPSConv2D ||
        stage.kind == GfxMpsrtStageKind::MPSGroupConv2D) {
        (void)gfx_mpsrt_read_conv2d_desc_attrs(module, prefix + ".conv2d", stage.conv2d_desc);
    }
    if (stage.kind == GfxMpsrtStageKind::MPSGemm) {
        (void)gfx_mpsrt_read_gemm_desc_attrs(module, prefix + ".gemm", stage.gemm_desc);
    }
    if (stage.kind == GfxMpsrtStageKind::MPSPool2D) {
        (void)gfx_mpsrt_read_pool2d_desc_attrs(module, prefix + ".pool2d", stage.pool2d_desc);
    }
    if (stage.kind == GfxMpsrtStageKind::MPSSoftmax) {
        (void)gfx_mpsrt_read_softmax_desc_attrs(module, prefix + ".softmax", stage.softmax_desc);
    }
    if (stage.kind == GfxMpsrtStageKind::MPSTopK) {
        (void)gfx_mpsrt_read_topk_desc_attrs(module, prefix + ".topk", stage.topk_desc);
    }
    if (stage.builder_symbol.empty()) {
        stage.builder_symbol = gfx_mpsrt_builder_symbol(stage.kind);
    }
    stage_record_key = gfx_mpsrt_stage_record_key(stage);
    return stage.domain != GfxStageBackendDomain::Unknown &&
           stage.kind != GfxMpsrtStageKind::Unknown &&
           gfx_mpsrt_stage_has_builder_symbol(stage.kind) &&
           !stage_record_key.empty();
}

}  // namespace detail

struct GfxMpsrtModuleStagePlan {
    bool valid = false;
    GfxMpsrtStageDesc stage{};
    std::vector<GfxMpsrtTensorDesc> inputs;
    std::vector<GfxMpsrtTensorDesc> outputs;
    std::string stage_record_key;
};

inline bool gfx_mpsrt_finalize_external_buffer_abi(GfxMpsrtExternalBufferAbiPlan& abi) {
    if (abi.has_buffer_roles) {
        for (const auto role : abi.buffer_roles) {
            if (!gfx_mpsrt_is_valid_external_buffer_role(role)) {
                return false;
            }
        }
    }
    if (abi.has_buffer_count && abi.has_output_buffer_count && abi.output_buffer_count > abi.buffer_count) {
        return false;
    }
    if (abi.has_buffer_roles && abi.has_buffer_count && abi.buffer_roles.size() != abi.buffer_count) {
        return false;
    }
    if (abi.has_buffer_roles && abi.has_output_buffer_count &&
        gfx_mpsrt_count_external_output_roles(abi.buffer_roles) != abi.output_buffer_count) {
        return false;
    }
    abi.valid = abi.has_buffer_count || abi.has_output_buffer_count || abi.has_buffer_roles;
    return abi.valid;
}

inline bool gfx_mpsrt_external_buffer_abi_from_kernel_manifest(mlir::ModuleOp module,
                                                               GfxMpsrtExternalBufferAbiPlan& abi,
                                                               uint32_t known_buffer_count = 0,
                                                               uint32_t known_output_buffer_count = 0) {
    abi = {};
    GfxKernelStageManifest manifest{};
    if (!detail::gfx_mpsrt_read_stage_manifest_attrs(module, manifest) ||
        !manifest.custom_kernel.valid ||
        !manifest.custom_kernel.external_buffer_abi.valid) {
        return false;
    }

    const auto& spec = manifest.custom_kernel.external_buffer_abi;
    if (!spec.roles.empty()) {
        abi.has_buffer_roles = true;
        abi.buffer_roles = gfx_mpsrt_external_buffer_roles_from_kernel_roles(spec.roles);
        if (abi.buffer_roles.size() == 2 &&
            abi.buffer_roles[0] == GfxMpsrtExternalBufferRole::TensorInput &&
            abi.buffer_roles[1] == GfxMpsrtExternalBufferRole::TensorOutput) {
            abi = {};
            return false;
        }
        abi.has_buffer_count = true;
        abi.buffer_count = static_cast<uint32_t>(abi.buffer_roles.size());
        abi.has_output_buffer_count = true;
        abi.output_buffer_count = gfx_mpsrt_count_external_output_roles(abi.buffer_roles);
        return gfx_mpsrt_finalize_external_buffer_abi(abi);
    }

    const uint32_t hinted_buffer_count = known_buffer_count;
    (void)known_output_buffer_count;

    if (spec.leading_input_count != 0 || spec.leading_output_count != 0) {
        const uint32_t leading_io_count = spec.leading_input_count + spec.leading_output_count;
        if (hinted_buffer_count == 0) {
            return false;
        }
        const uint32_t buffer_count = hinted_buffer_count;
        if (buffer_count < leading_io_count) {
            return false;
        }
        abi.has_buffer_count = true;
        abi.buffer_count = buffer_count;
        abi.has_output_buffer_count = true;
        abi.output_buffer_count = spec.leading_output_count;
        abi.has_buffer_roles = true;
        abi.buffer_roles = gfx_mpsrt_external_buffer_roles_from_leading_io_spec(spec, buffer_count);
        return gfx_mpsrt_finalize_external_buffer_abi(abi);
    }

    return false;
}

inline GfxMpsrtExternalBufferAbiPlan read_module_mpsrt_external_buffer_abi(mlir::ModuleOp module) {
    GfxMpsrtExternalBufferAbiPlan abi{};
    if (!module) {
        return abi;
    }

    if (gfx_mpsrt_external_buffer_abi_from_kernel_manifest(module, abi)) {
        return abi;
    }

    return abi;
}

inline bool read_module_mpsrt_stage_plan(mlir::ModuleOp module,
                                         GfxMpsrtModuleStagePlan& out) {
    out = {};
    if (!module) {
        return false;
    }

    GfxMpsrtProgram ops_program{};
    if (read_module_mpsrt_ops_program(module, ops_program)) {
        if (ops_program.multi_stage || ops_program.stages.size() != 1) {
            return false;
        }
        const auto& stage_spec = ops_program.stages.front();
        out.stage = stage_spec.stage;
        out.inputs = ops_program.inputs;
        out.outputs = stage_spec.output_descs;
        out.stage_record_key = stage_spec.stage_record_key;
        out.valid = true;
        return true;
    }

    const bool has_stage_manifest =
        detail::gfx_mpsrt_read_stage_manifest_attrs(module, out.stage.stage_manifest);
    if (!has_stage_manifest) {
        return false;
    }

    detail::gfx_mpsrt_apply_stage_manifest_to_stage_desc(out.stage);
    if (out.stage.kind == GfxMpsrtStageKind::MPSGemm) {
        (void)detail::gfx_mpsrt_read_gemm_desc_attrs(module, out.stage.gemm_desc);
    }
    if (out.stage.kind == GfxMpsrtStageKind::MPSConv2D ||
        out.stage.kind == GfxMpsrtStageKind::MPSGroupConv2D) {
        (void)detail::gfx_mpsrt_read_conv2d_desc_attrs(module, out.stage.conv2d_desc);
    }
    if (out.stage.kind == GfxMpsrtStageKind::MPSPool2D) {
        (void)detail::gfx_mpsrt_read_pool2d_desc_attrs(module, out.stage.pool2d_desc);
    }
    if (out.stage.kind == GfxMpsrtStageKind::MPSSoftmax) {
        (void)detail::gfx_mpsrt_read_softmax_desc_attrs(module, out.stage.softmax_desc);
    }
    if (out.stage.kind == GfxMpsrtStageKind::MPSTopK) {
        (void)detail::gfx_mpsrt_read_topk_desc_attrs(module, out.stage.topk_desc);
    }

    if (out.stage.stage_type.empty()) {
        out.stage.stage_type = gfx_mpsrt_stage_type_from_manifest(out.stage.stage_manifest);
    }
    out.stage.input_storage = !out.inputs.empty() ? out.inputs.front().storage : GfxMpsrtStorage::Unknown;
    out.stage.output_storage = !out.outputs.empty() ? out.outputs.front().storage : out.stage.input_storage;
    out.stage.layout = out.stage.output_storage != GfxMpsrtStorage::Unknown
                           ? gfx_mpsrt_stage_layout_for_storage(out.stage.output_storage)
                           : GfxMpsrtLayout::Unknown;
    if (out.stage.kernel_name.empty()) {
        out.stage.kernel_name = !out.stage.dispatch_entry_point.empty()
                                    ? out.stage.dispatch_entry_point
                                    : gfx_mpsrt_default_kernel_name(out.stage.kind, out.stage.stage_type);
    }
    if (out.stage.builder_symbol.empty()) {
        out.stage.builder_symbol = gfx_mpsrt_builder_symbol(out.stage.kind);
    }
    out.stage_record_key = gfx_mpsrt_stage_record_key(out.stage);
    out.valid = out.stage.domain != GfxStageBackendDomain::Unknown &&
                out.stage.kind != GfxMpsrtStageKind::Unknown &&
                gfx_mpsrt_stage_has_builder_symbol(out.stage.kind) &&
                !out.stage_record_key.empty();
    return out.valid;
}

inline constexpr const char* kGfxAppleStorageAssignmentPrefix =
    "gfx.apple.pipeline.storage_assignment";

inline bool build_mpsrt_storage_assignment_from_stage_plan(
    const GfxMpsrtModuleStagePlan& stage_plan,
    std::vector<GfxMpsrtStorageBridgeDesc>& bridges) {
    bridges.clear();
    if (!stage_plan.valid) {
        return false;
    }
    const auto builder_plan = gfx_mpsrt_make_builder_plan(stage_plan.stage,
                                                          stage_plan.inputs,
                                                          stage_plan.outputs,
                                                          stage_plan.stage_record_key);
    if (!builder_plan.valid) {
        return false;
    }
    bridges = builder_plan.storage_bridges;
    return true;
}

inline void erase_module_apple_storage_assignment_attrs(mlir::ModuleOp module) {
    if (!module) {
        return;
    }
    std::vector<std::string> attrs_to_remove;
    const llvm::StringRef prefix = kGfxAppleStorageAssignmentPrefix;
    for (const auto& named_attr : module->getAttrs()) {
        const auto name = named_attr.getName().strref();
        if (name.starts_with(prefix)) {
            attrs_to_remove.push_back(name.str());
        }
    }
    for (const auto& name : attrs_to_remove) {
        module->removeAttr(name);
    }
}

inline bool annotate_module_with_apple_storage_assignment(
    mlir::ModuleOp module,
    const GfxMpsrtModuleStagePlan& stage_plan,
    std::vector<GfxMpsrtStorageBridgeDesc>* bridges_out = nullptr) {
    if (!module) {
        return false;
    }
    std::vector<GfxMpsrtStorageBridgeDesc> bridges;
    if (!build_mpsrt_storage_assignment_from_stage_plan(stage_plan, bridges)) {
        return false;
    }
    erase_module_apple_storage_assignment_attrs(module);
    detail::gfx_mpsrt_set_storage_bridges_attrs(module,
                                                kGfxAppleStorageAssignmentPrefix,
                                                bridges);
    if (bridges_out) {
        *bridges_out = bridges;
    }
    return true;
}

inline void annotate_module_with_empty_apple_storage_assignment(mlir::ModuleOp module) {
    if (!module) {
        return;
    }
    erase_module_apple_storage_assignment_attrs(module);
    detail::gfx_mpsrt_set_storage_bridges_attrs(module,
                                                kGfxAppleStorageAssignmentPrefix,
                                                {});
}

inline bool read_module_apple_storage_assignment(
    mlir::ModuleOp module,
    std::vector<GfxMpsrtStorageBridgeDesc>& bridges) {
    bridges.clear();
    if (!module) {
        return false;
    }
    return detail::gfx_mpsrt_read_storage_bridges_attrs(module,
                                                       kGfxAppleStorageAssignmentPrefix,
                                                       bridges);
}

inline bool materialize_module_mpsrt_ops_from_stage_plan(mlir::ModuleOp module,
                                                         const GfxMpsrtModuleStagePlan& stage_plan,
                                                         const GfxMpsrtExternalBufferAbiPlan& external_buffer_abi = {}) {
    if (!module || !stage_plan.valid) {
        return false;
    }

    GfxMpsrtBuilderStageSpec stage{};
    stage.stage = stage_plan.stage;
    stage.stage_record_key = stage_plan.stage_record_key;
    stage.inputs = gfx_mpsrt_make_sequential_values(stage_plan.inputs.size());
    stage.outputs = gfx_mpsrt_make_sequential_values(stage_plan.outputs.size(),
                                                     static_cast<GfxMpsrtValue>(stage_plan.inputs.size()));
    stage.output_descs = stage_plan.outputs;

    GfxMpsrtProgram program{};
    program.valid = true;
    program.multi_stage = false;
    program.record_key = stage_plan.stage_record_key;
    program.inputs = stage_plan.inputs;
    program.output_values = stage.outputs;
    program.stages.push_back(std::move(stage));
    program.external_buffer_abi = external_buffer_abi.valid
                                      ? external_buffer_abi
                                      : read_module_mpsrt_external_buffer_abi(module);

    std::vector<GfxMpsrtStorageBridgeDesc> storage_assignment_bridges;
    if (read_module_apple_storage_assignment(module, storage_assignment_bridges) ||
        build_mpsrt_storage_assignment_from_stage_plan(stage_plan, storage_assignment_bridges)) {
        program.has_storage_bridges = true;
        program.storage_bridges = std::move(storage_assignment_bridges);
    }
    if (!materialize_module_mpsrt_ops(module, program)) {
        return false;
    }
    erase_module_mpsrt_legacy_attrs(module);
    return true;
}

inline bool read_module_mpsrt_program(mlir::ModuleOp module,
                                      GfxMpsrtProgram& out) {
    out = {};
    if (!module) {
        return false;
    }
    if (read_module_mpsrt_ops_program(module, out)) {
        return true;
    }

    GfxMpsrtModuleStagePlan stage_plan{};
    if (!read_module_mpsrt_stage_plan(module, stage_plan)) {
        return false;
    }
    out.multi_stage = false;
    out.record_key = stage_plan.stage_record_key;
    out.inputs = std::move(stage_plan.inputs);
    GfxMpsrtBuilderStageSpec stage{};
    stage.stage = std::move(stage_plan.stage);
    stage.stage_record_key = stage_plan.stage_record_key;
    stage.inputs = gfx_mpsrt_make_sequential_values(out.inputs.size());
    stage.outputs = gfx_mpsrt_make_sequential_values(stage_plan.outputs.size(),
                                                     static_cast<GfxMpsrtValue>(out.inputs.size()));
    stage.output_descs = std::move(stage_plan.outputs);
    out.output_values = stage.outputs;
    out.stages.push_back(std::move(stage));

    out.external_buffer_abi = read_module_mpsrt_external_buffer_abi(module);
    if (detail::gfx_mpsrt_read_storage_bridges_attrs(module, out.storage_bridges)) {
        out.has_storage_bridges = true;
    }

    GfxMpsrtBuilderPlan builder_plan{};
    out.valid = gfx_mpsrt_build_builder_plan_from_program(out, builder_plan);
    return out.valid;
}

struct GfxMpsrtModuleBuilderPlan {
    bool valid = false;
    GfxMpsrtProgram program{};
    GfxMpsrtModuleStagePlan stage_plan{};
    bool multi_stage = false;
    GfxMpsrtExternalBufferAbiPlan external_buffer_abi{};
    GfxMpsrtBuilderPlan builder_plan{};
};

inline bool build_module_mpsrt_builder_plan(mlir::ModuleOp module,
                                            GfxMpsrtModuleBuilderPlan& out) {
    out = {};
    GfxMpsrtProgram program{};
    if (!read_module_mpsrt_program(module, program) ||
        !gfx_mpsrt_build_builder_plan_from_program(program, out.builder_plan)) {
        out = {};
        return false;
    }

    out.program = program;
    out.multi_stage = program.multi_stage;
    out.external_buffer_abi = program.external_buffer_abi;
    out.stage_plan.valid = true;
    out.stage_plan.stage = program.stages.front().stage;
    out.stage_plan.inputs = program.inputs;
    out.stage_plan.outputs = program.stages.back().output_descs;
    out.stage_plan.stage_record_key = program.stages.front().stage_record_key;

    out.valid = out.builder_plan.valid;
    return out.valid;
}

inline GfxMpsrtModuleBuilderPlan build_module_mpsrt_builder_plan(mlir::ModuleOp module) {
    GfxMpsrtModuleBuilderPlan out{};
    (void)build_module_mpsrt_builder_plan(module, out);
    return out;
}

inline bool build_module_mpsrt_stage_plan(mlir::ModuleOp module,
                                          const GfxStageOptimizationPlan& plan,
                                          const std::string& stage_type,
                                          std::string_view kernel_entry_point,
                                          GfxMpsrtModuleStagePlan& stage_plan,
                                          const std::vector<GfxKernelBufferRole>& semantic_input_roles = {}) {
    stage_plan = {};
    if (!module) {
        return false;
    }

    stage_plan.stage = gfx_mpsrt_make_stage_desc(plan, stage_type, kernel_entry_point);
    stage_plan.stage_record_key = gfx_mpsrt_stage_record_key(stage_plan.stage);
    if (!detail::gfx_mpsrt_collect_entry_tensors(module,
                                                plan,
                                                stage_plan.inputs,
                                                stage_plan.outputs,
                                                semantic_input_roles)) {
        return false;
    }
    detail::gfx_mpsrt_apply_semantic_input_roles(stage_plan.inputs, semantic_input_roles);
    if (stage_plan.stage.stage_manifest.valid) {
        stage_plan.stage.stage_manifest.semantic_input_roles =
            detail::gfx_mpsrt_normalize_semantic_input_roles(stage_plan.inputs.size(),
                                                             semantic_input_roles);
        stage_plan.stage.stage_manifest.semantic_output_roles =
            detail::gfx_mpsrt_default_semantic_output_roles(stage_plan.outputs.size());
    }
    stage_plan.valid = stage_plan.stage.domain != GfxStageBackendDomain::Unknown &&
                       stage_plan.stage.kind != GfxMpsrtStageKind::Unknown &&
                       gfx_mpsrt_stage_has_builder_symbol(stage_plan.stage.kind) &&
                       !stage_plan.stage_record_key.empty();
    if (!stage_plan.valid) {
        return false;
    }
    return true;
}

inline bool annotate_module_with_mpsrt_stage_manifest(mlir::ModuleOp module,
                                                      const GfxStageOptimizationPlan& plan,
                                                      const std::string& stage_type,
                                                      std::string_view kernel_entry_point,
                                                      GfxMpsrtModuleStagePlan& stage_plan,
                                                      const std::vector<GfxKernelBufferRole>& semantic_input_roles = {}) {
    if (!build_module_mpsrt_stage_plan(module,
                                       plan,
                                       stage_type,
                                       kernel_entry_point,
                                       stage_plan,
                                       semantic_input_roles)) {
        return false;
    }
    detail::gfx_mpsrt_set_stage_manifest_attrs(module, stage_plan.stage.stage_manifest);
    return true;
}

struct GfxAppleMpsStageLoweringPlan {
    bool valid = false;
    GfxMpsrtModuleStagePlan stage_plan;
};

inline bool gfx_mpsrt_stage_is_apple_mps_vendor(const GfxMpsrtStageDesc& stage) {
    return stage.domain == GfxStageBackendDomain::AppleMps &&
           stage.uses_vendor_primitive &&
           !stage.uses_custom_kernel &&
           stage.stage_manifest.valid &&
           stage.stage_manifest.backend_domain == GfxKernelBackendDomain::AppleMps &&
           stage.stage_manifest.execution_kind == GfxKernelExecutionKind::VendorPrimitive;
}

inline bool refresh_apple_mps_stage_record_key(GfxAppleMpsStageLoweringPlan& lowering_plan) {
    if (!lowering_plan.valid ||
        !gfx_mpsrt_stage_is_apple_mps_vendor(lowering_plan.stage_plan.stage)) {
        return false;
    }
    lowering_plan.stage_plan.stage_record_key =
        gfx_mpsrt_stage_record_key(lowering_plan.stage_plan.stage);
    lowering_plan.stage_plan.valid = !lowering_plan.stage_plan.stage_record_key.empty();
    lowering_plan.valid = lowering_plan.stage_plan.valid;
    return lowering_plan.valid;
}

inline bool materialize_apple_mps_typed_program(
    mlir::ModuleOp module,
    const GfxAppleMpsStageLoweringPlan& lowering_plan);

inline bool materialize_apple_mps_typed_program(
    mlir::ModuleOp module,
    const GfxAppleMpsStageLoweringPlan& lowering_plan) {
    if (!module || !lowering_plan.valid ||
        !gfx_mpsrt_stage_is_apple_mps_vendor(lowering_plan.stage_plan.stage)) {
        return false;
    }
    return materialize_module_mpsrt_ops_from_stage_plan(module, lowering_plan.stage_plan);
}

}  // namespace gfx_plugin
}  // namespace ov
