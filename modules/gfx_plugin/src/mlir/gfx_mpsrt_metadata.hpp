// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "openvino/core/type/element_type.hpp"
#include "runtime/gfx_mpsrt_abi.hpp"
#include "runtime/gfx_mpsrt_builder_plan.hpp"
#include "runtime/gfx_mpsrt_plan.hpp"
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

inline bool gfx_mpsrt_read_i32_attr(mlir::ModuleOp module, const std::string& name, uint32_t& out) {
    auto attr = module->getAttrOfType<mlir::IntegerAttr>(name);
    if (!attr) {
        return false;
    }
    out = static_cast<uint32_t>(attr.getInt());
    return true;
}

inline bool gfx_mpsrt_read_i64_attr(mlir::ModuleOp module, const std::string& name, uint64_t& out) {
    auto attr = module->getAttrOfType<mlir::IntegerAttr>(name);
    if (!attr) {
        return false;
    }
    out = static_cast<uint64_t>(attr.getInt());
    return true;
}

inline bool gfx_mpsrt_read_string_attr(mlir::ModuleOp module, const std::string& name, std::string& out) {
    auto attr = module->getAttrOfType<mlir::StringAttr>(name);
    if (!attr) {
        return false;
    }
    out = attr.str();
    return true;
}

inline bool gfx_mpsrt_read_bool_attr(mlir::ModuleOp module, const std::string& name, bool& out) {
    auto attr = module->getAttrOfType<mlir::BoolAttr>(name);
    if (!attr) {
        return false;
    }
    out = attr.getValue();
    return true;
}

inline void gfx_mpsrt_read_u64_array_attr(mlir::ModuleOp module,
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

inline void gfx_mpsrt_read_i64_array_attr(mlir::ModuleOp module,
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

inline std::vector<uint32_t> gfx_mpsrt_read_u32_vector_attr(mlir::ModuleOp module,
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

inline void gfx_mpsrt_set_tensor_desc_attrs(mlir::ModuleOp module,
                                            std::string prefix,
                                            const GfxMpsrtTensorDesc& desc) {
    mlir::Builder builder(module.getContext());
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

inline bool gfx_mpsrt_read_tensor_desc_attrs(mlir::ModuleOp module,
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

inline void gfx_mpsrt_annotate_entry_tensors(mlir::ModuleOp module,
                                             const GfxStageOptimizationPlan& plan) {
    auto func = gfx_mpsrt_entry_func(module);
    if (!func) {
        return;
    }

    auto fn_type = func.getFunctionType();
    mlir::Builder builder(module.getContext());
    module->setAttr("gfx.mpsrt.input_count",
                    builder.getI32IntegerAttr(static_cast<int32_t>(fn_type.getNumInputs())));
    module->setAttr("gfx.mpsrt.output_count",
                    builder.getI32IntegerAttr(static_cast<int32_t>(fn_type.getNumResults())));

    const auto storage = plan.placement.storage;
    for (unsigned i = 0; i < fn_type.getNumInputs(); ++i) {
        const auto type = fn_type.getInput(i);
        auto desc = gfx_mpsrt_make_tensor_desc(gfx_mpsrt_shape_from_mlir_type(type),
                                               gfx_mpsrt_element_from_mlir_type(type),
                                               storage,
                                               GfxMpsrtTensorFlagExternalIo);
        gfx_mpsrt_set_tensor_desc_attrs(module, "gfx.mpsrt.input" + std::to_string(i), desc);
    }
    for (unsigned i = 0; i < fn_type.getNumResults(); ++i) {
        const auto type = fn_type.getResult(i);
        auto desc = gfx_mpsrt_make_tensor_desc(gfx_mpsrt_shape_from_mlir_type(type),
                                               gfx_mpsrt_element_from_mlir_type(type),
                                               storage,
                                               GfxMpsrtTensorFlagTransient);
        gfx_mpsrt_set_tensor_desc_attrs(module, "gfx.mpsrt.output" + std::to_string(i), desc);
    }
}

}  // namespace detail

struct GfxMpsrtModuleStagePlan {
    bool valid = false;
    GfxMpsrtStageDesc stage{};
    std::vector<GfxMpsrtTensorDesc> inputs;
    std::vector<GfxMpsrtTensorDesc> outputs;
    std::string stage_record_key;
};

struct GfxMpsrtExternalBufferAbiPlan {
    bool valid = false;
    bool has_buffer_count = false;
    bool has_output_buffer_count = false;
    bool has_buffer_roles = false;
    uint32_t buffer_count = 0;
    uint32_t output_buffer_count = 0;
    std::vector<GfxMpsrtExternalBufferRole> buffer_roles;
};

inline GfxMpsrtExternalBufferAbiPlan read_module_mpsrt_external_buffer_abi(mlir::ModuleOp module) {
    GfxMpsrtExternalBufferAbiPlan abi{};
    if (!module) {
        return abi;
    }

    if (detail::gfx_mpsrt_read_i32_attr(module, "gfx.mpsrt.external_buffer_count", abi.buffer_count)) {
        abi.has_buffer_count = true;
    }

    if (detail::gfx_mpsrt_read_i32_attr(module,
                                        "gfx.mpsrt.external_output_buffer_count",
                                        abi.output_buffer_count)) {
        abi.has_output_buffer_count = true;
    }
    const auto role_values = detail::gfx_mpsrt_read_u32_vector_attr(module, "gfx.mpsrt.external_buffer_roles");
    if (!role_values.empty()) {
        abi.has_buffer_roles = true;
        abi.buffer_roles.reserve(role_values.size());
        for (const auto role_value : role_values) {
            abi.buffer_roles.push_back(static_cast<GfxMpsrtExternalBufferRole>(role_value));
        }
    }
    if (abi.has_buffer_count && abi.has_output_buffer_count && abi.output_buffer_count > abi.buffer_count) {
        return abi;
    }
    if (abi.has_buffer_roles && abi.has_buffer_count && abi.buffer_roles.size() != abi.buffer_count) {
        return abi;
    }
    if (abi.has_buffer_roles && abi.has_output_buffer_count) {
        uint32_t role_output_count = 0;
        for (const auto role : abi.buffer_roles) {
            if (role == GfxMpsrtExternalBufferRole::TensorOutput) {
                ++role_output_count;
            }
        }
        if (role_output_count != abi.output_buffer_count) {
            return abi;
        }
    }
    abi.valid = abi.has_buffer_count || abi.has_output_buffer_count || abi.has_buffer_roles;
    return abi;
}

inline bool read_module_mpsrt_stage_plan(mlir::ModuleOp module,
                                         GfxMpsrtModuleStagePlan& out) {
    out = {};
    if (!module) {
        return false;
    }

    std::string backend;
    std::string stage_kind;
    std::string input_storage;
    std::string output_storage;
    std::string layout;
    if (!detail::gfx_mpsrt_read_string_attr(module, "gfx.backend", backend) ||
        !detail::gfx_mpsrt_read_string_attr(module, "gfx.mpsrt.stage_kind", stage_kind) ||
        !detail::gfx_mpsrt_read_string_attr(module, "gfx.mpsrt.stage_record_key", out.stage_record_key)) {
        return false;
    }

    out.stage.domain = detail::gfx_mpsrt_backend_domain_from_name(backend);
    out.stage.kind = gfx_mpsrt_stage_kind_from_name(stage_kind);
    (void)detail::gfx_mpsrt_read_string_attr(module, "gfx.stage_type", out.stage.stage_type);
    (void)detail::gfx_mpsrt_read_string_attr(module, "gfx.mpsrt.kernel_name", out.stage.kernel_name);
    (void)detail::gfx_mpsrt_read_string_attr(module, "gfx.mpsrt.builder_symbol", out.stage.builder_symbol);
    (void)detail::gfx_mpsrt_read_string_attr(module, "gfx.specialization_key", out.stage.specialization_key);
    (void)detail::gfx_mpsrt_read_string_attr(module,
                                             "gfx.mpsrt.dispatch_kernel_family",
                                             out.stage.dispatch_kernel_family);
    (void)detail::gfx_mpsrt_read_string_attr(module,
                                             "gfx.mpsrt.dispatch_entry_point",
                                             out.stage.dispatch_entry_point);
    (void)detail::gfx_mpsrt_read_i32_attr(module,
                                          "gfx.mpsrt.dispatch_kernel_family_id",
                                          out.stage.dispatch_kernel_family_id);
    (void)detail::gfx_mpsrt_read_i32_attr(module,
                                          "gfx.mpsrt.dispatch_flags",
                                          out.stage.dispatch_flags);
    (void)detail::gfx_mpsrt_read_i32_attr(module,
                                          "gfx.mpsrt.dispatch_threads_per_threadgroup",
                                          out.stage.dispatch_threads_per_threadgroup);
    (void)detail::gfx_mpsrt_read_bool_attr(module,
                                           "gfx.mpsrt.dispatch_precompiled_kernel_required",
                                           out.stage.dispatch_precompiled_kernel_required);
    (void)detail::gfx_mpsrt_read_bool_attr(module, "gfx.uses_vendor_primitive", out.stage.uses_vendor_primitive);
    (void)detail::gfx_mpsrt_read_bool_attr(module, "gfx.uses_custom_kernel", out.stage.uses_custom_kernel);

    uint32_t input_count = 0;
    uint32_t output_count = 0;
    (void)detail::gfx_mpsrt_read_i32_attr(module, "gfx.mpsrt.input_count", input_count);
    (void)detail::gfx_mpsrt_read_i32_attr(module, "gfx.mpsrt.output_count", output_count);
    out.inputs.reserve(input_count);
    out.outputs.reserve(output_count);
    for (uint32_t i = 0; i < input_count; ++i) {
        GfxMpsrtTensorDesc desc{};
        if (detail::gfx_mpsrt_read_tensor_desc_attrs(module, "gfx.mpsrt.input" + std::to_string(i), desc)) {
            out.inputs.push_back(desc);
        }
    }
    for (uint32_t i = 0; i < output_count; ++i) {
        GfxMpsrtTensorDesc desc{};
        if (detail::gfx_mpsrt_read_tensor_desc_attrs(module, "gfx.mpsrt.output" + std::to_string(i), desc)) {
            out.outputs.push_back(desc);
        }
    }

    out.stage.input_storage = !out.inputs.empty() ? out.inputs.front().storage : GfxMpsrtStorage::Unknown;
    out.stage.output_storage = !out.outputs.empty() ? out.outputs.front().storage : out.stage.input_storage;
    out.stage.layout = out.stage.output_storage != GfxMpsrtStorage::Unknown
                           ? gfx_mpsrt_stage_layout_for_storage(out.stage.output_storage)
                           : GfxMpsrtLayout::Unknown;
    if (out.stage.builder_symbol.empty()) {
        out.stage.builder_symbol = gfx_mpsrt_builder_symbol(out.stage.kind);
    }
    out.valid = out.stage.domain != GfxStageBackendDomain::Unknown &&
                out.stage.kind != GfxMpsrtStageKind::Unknown &&
                gfx_mpsrt_stage_has_builder_symbol(out.stage.kind) &&
                !out.stage_record_key.empty();
    return out.valid;
}

struct GfxMpsrtModuleBuilderPlan {
    bool valid = false;
    GfxMpsrtModuleStagePlan stage_plan{};
    GfxMpsrtExternalBufferAbiPlan external_buffer_abi{};
    GfxMpsrtBuilderPlan builder_plan{};
};

inline bool build_module_mpsrt_builder_plan(mlir::ModuleOp module,
                                            GfxMpsrtModuleBuilderPlan& out) {
    out = {};
    if (!read_module_mpsrt_stage_plan(module, out.stage_plan)) {
        return false;
    }

    out.builder_plan = gfx_mpsrt_make_builder_plan(out.stage_plan.stage,
                                                   out.stage_plan.inputs,
                                                   out.stage_plan.outputs,
                                                   out.stage_plan.stage_record_key);
    out.external_buffer_abi = read_module_mpsrt_external_buffer_abi(module);
    if (out.external_buffer_abi.valid) {
        out.builder_plan.external_buffer_abi_valid = true;
        out.builder_plan.external_buffer_count = out.external_buffer_abi.buffer_count;
        out.builder_plan.external_output_buffer_count = out.external_buffer_abi.output_buffer_count;
        out.builder_plan.external_buffer_roles = out.external_buffer_abi.buffer_roles;
    }
    out.valid = out.stage_plan.valid && out.builder_plan.valid;
    return out.valid;
}

inline GfxMpsrtModuleBuilderPlan build_module_mpsrt_builder_plan(mlir::ModuleOp module) {
    GfxMpsrtModuleBuilderPlan out{};
    (void)build_module_mpsrt_builder_plan(module, out);
    return out;
}

inline void annotate_module_with_mpsrt_stage_plan(mlir::ModuleOp module,
                                                  const GfxStageOptimizationPlan& plan,
                                                  const std::string& stage_type) {
    if (!module) {
        return;
    }

    mlir::Builder builder(module.getContext());
    module->setAttr("gfx.backend",
                    builder.getStringAttr(gfx_stage_backend_domain_name(plan.placement.domain)));
    module->setAttr("gfx.storage",
                    builder.getStringAttr(gfx_stage_storage_kind_name(plan.placement.storage)));
    module->setAttr("gfx.uses_vendor_primitive",
                    builder.getBoolAttr(plan.placement.uses_vendor_primitive));
    module->setAttr("gfx.uses_custom_kernel",
                    builder.getBoolAttr(plan.placement.uses_custom_kernel));
    if (!plan.placement.specialization_key.empty()) {
        module->setAttr("gfx.specialization_key",
                        builder.getStringAttr(plan.placement.specialization_key));
    }
    if (!stage_type.empty()) {
        module->setAttr("gfx.stage_type", builder.getStringAttr(stage_type));
    }
    const auto stage_desc = gfx_mpsrt_make_stage_desc(plan, stage_type);
    module->setAttr("gfx.mpsrt.stage_kind",
                    builder.getStringAttr(gfx_mpsrt_stage_kind_name(stage_desc.kind)));
    module->setAttr("gfx.mpsrt.kernel_name", builder.getStringAttr(stage_desc.kernel_name));
    module->setAttr("gfx.mpsrt.builder_symbol", builder.getStringAttr(stage_desc.builder_symbol));
    if (!stage_desc.dispatch_kernel_family.empty()) {
        module->setAttr("gfx.mpsrt.dispatch_kernel_family",
                        builder.getStringAttr(stage_desc.dispatch_kernel_family));
    }
    if (!stage_desc.dispatch_entry_point.empty()) {
        module->setAttr("gfx.mpsrt.dispatch_entry_point",
                        builder.getStringAttr(stage_desc.dispatch_entry_point));
    }
    if (stage_desc.dispatch_kernel_family_id != 0) {
        module->setAttr("gfx.mpsrt.dispatch_kernel_family_id",
                        builder.getI32IntegerAttr(static_cast<int32_t>(stage_desc.dispatch_kernel_family_id)));
    }
    if (stage_desc.dispatch_flags != GfxMpsrtMslDispatchFlagNone) {
        module->setAttr("gfx.mpsrt.dispatch_flags",
                        builder.getI32IntegerAttr(static_cast<int32_t>(stage_desc.dispatch_flags)));
    }
    if (stage_desc.dispatch_threads_per_threadgroup != 0) {
        module->setAttr("gfx.mpsrt.dispatch_threads_per_threadgroup",
                        builder.getI32IntegerAttr(static_cast<int32_t>(stage_desc.dispatch_threads_per_threadgroup)));
    }
    if (stage_desc.dispatch_precompiled_kernel_required) {
        module->setAttr("gfx.mpsrt.dispatch_precompiled_kernel_required", builder.getBoolAttr(true));
    }
    module->setAttr("gfx.mpsrt.stage_record_key",
                    builder.getStringAttr(gfx_mpsrt_stage_record_key(stage_desc)));
    detail::gfx_mpsrt_annotate_entry_tensors(module, plan);
}

}  // namespace gfx_plugin
}  // namespace ov
