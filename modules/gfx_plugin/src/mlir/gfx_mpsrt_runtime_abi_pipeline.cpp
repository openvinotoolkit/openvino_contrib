// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/gfx_mpsrt_runtime_abi_pipeline.hpp"

#include "mlir/gfx_mpsrt_metadata.hpp"
#include "mlir/gfx_mpsrt_ops.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace ov {
namespace gfx_plugin {
namespace {

constexpr const char* kRuntimeAbiPlanSymbol = "gfx_mpsrt_runtime_abi_plan";

struct GfxMpsrtRuntimeAbiProgramPlan {
    bool valid = false;
    GfxMpsrtProgram program{};
    GfxMpsrtBuilderPlan builder_plan{};
};

void ensure_runtime_abi_dialects(mlir::ModuleOp module) {
    if (!module) {
        return;
    }
    module.getContext()->loadDialect<mlir::func::FuncDialect>();
}

GfxMpsrtBuilderRecordKind builder_record_kind_from_name(llvm::StringRef name) {
    if (name == "model_begin") return GfxMpsrtBuilderRecordKind::ModelBegin;
    if (name == "add_tensor") return GfxMpsrtBuilderRecordKind::AddTensor;
    if (name == "encode_stage") return GfxMpsrtBuilderRecordKind::EncodeStage;
    if (name == "model_end") return GfxMpsrtBuilderRecordKind::ModelEnd;
    return GfxMpsrtBuilderRecordKind::Unknown;
}

std::vector<uint32_t> read_u32_vector_attr(mlir::Operation* op, llvm::StringRef name) {
    std::vector<uint32_t> values;
    if (!op) {
        return values;
    }
    auto attr = op->getAttrOfType<mlir::ArrayAttr>(name);
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

void set_u32_vector_attr(mlir::Operation* op,
                         mlir::Builder& builder,
                         llvm::StringRef name,
                         const std::vector<uint32_t>& values) {
    if (!values.empty()) {
        op->setAttr(name, detail::gfx_mpsrt_u32_vector_attr(builder, values));
    }
}

void set_u64_array_attr(mlir::Operation* op,
                        mlir::Builder& builder,
                        const std::string& name,
                        const uint64_t* values,
                        uint32_t count) {
    llvm::SmallVector<mlir::Attribute, 8> attrs;
    attrs.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        attrs.push_back(builder.getI64IntegerAttr(static_cast<int64_t>(values[i])));
    }
    op->setAttr(name, builder.getArrayAttr(attrs));
}

void set_i64_array_attr(mlir::Operation* op,
                        mlir::Builder& builder,
                        const std::string& name,
                        const int64_t* values,
                        uint32_t count) {
    llvm::SmallVector<mlir::Attribute, 8> attrs;
    attrs.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        attrs.push_back(builder.getI64IntegerAttr(values[i]));
    }
    op->setAttr(name, builder.getArrayAttr(attrs));
}

bool read_u32_attr(mlir::Operation* op, const std::string& name, uint32_t& out) {
    if (!op) {
        return false;
    }
    auto attr = op->getAttrOfType<mlir::IntegerAttr>(name);
    if (!attr) {
        return false;
    }
    out = static_cast<uint32_t>(attr.getInt());
    return true;
}

bool read_u64_attr(mlir::Operation* op, const std::string& name, uint64_t& out) {
    if (!op) {
        return false;
    }
    auto attr = op->getAttrOfType<mlir::IntegerAttr>(name);
    if (!attr) {
        return false;
    }
    out = static_cast<uint64_t>(attr.getInt());
    return true;
}

bool read_bool_attr(mlir::Operation* op, const std::string& name, bool& out) {
    if (!op) {
        return false;
    }
    auto attr = op->getAttrOfType<mlir::BoolAttr>(name);
    if (!attr) {
        return false;
    }
    out = attr.getValue();
    return true;
}

void read_u64_array_attr(mlir::Operation* op,
                         const std::string& name,
                         uint64_t* out,
                         uint32_t count) {
    if (!op) {
        return;
    }
    auto attr = op->getAttrOfType<mlir::ArrayAttr>(name);
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

void read_i64_array_attr(mlir::Operation* op,
                         const std::string& name,
                         int64_t* out,
                         uint32_t count) {
    if (!op) {
        return;
    }
    auto attr = op->getAttrOfType<mlir::ArrayAttr>(name);
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

void set_tensor_desc_attrs(mlir::Operation* op,
                           mlir::Builder& builder,
                           const std::string& prefix,
                           const GfxMpsrtTensorAbiDesc& desc) {
    op->setAttr(prefix + ".rank", builder.getI32IntegerAttr(static_cast<int32_t>(desc.rank)));
    op->setAttr(prefix + ".dtype", builder.getI32IntegerAttr(static_cast<int32_t>(desc.dtype)));
    op->setAttr(prefix + ".storage", builder.getI32IntegerAttr(static_cast<int32_t>(desc.storage)));
    op->setAttr(prefix + ".layout", builder.getI32IntegerAttr(static_cast<int32_t>(desc.layout)));
    op->setAttr(prefix + ".flags", builder.getI32IntegerAttr(static_cast<int32_t>(desc.flags)));
    op->setAttr(prefix + ".byte_offset", builder.getI64IntegerAttr(static_cast<int64_t>(desc.byte_offset)));
    op->setAttr(prefix + ".byte_length", builder.getI64IntegerAttr(static_cast<int64_t>(desc.byte_length)));
    set_u64_array_attr(op, builder, prefix + ".dims", desc.dims, std::min<uint32_t>(desc.rank, 8u));
    set_i64_array_attr(op, builder, prefix + ".strides", desc.strides, std::min<uint32_t>(desc.rank, 8u));
    op->setAttr(prefix + ".image_width", builder.getI32IntegerAttr(static_cast<int32_t>(desc.image_width)));
    op->setAttr(prefix + ".image_height", builder.getI32IntegerAttr(static_cast<int32_t>(desc.image_height)));
    op->setAttr(prefix + ".image_feature_channels",
                builder.getI32IntegerAttr(static_cast<int32_t>(desc.image_feature_channels)));
    op->setAttr(prefix + ".image_batch", builder.getI32IntegerAttr(static_cast<int32_t>(desc.image_batch)));
    op->setAttr(prefix + ".matrix_rows", builder.getI32IntegerAttr(static_cast<int32_t>(desc.matrix_rows)));
    op->setAttr(prefix + ".matrix_columns", builder.getI32IntegerAttr(static_cast<int32_t>(desc.matrix_columns)));
    op->setAttr(prefix + ".matrix_row_bytes", builder.getI32IntegerAttr(static_cast<int32_t>(desc.matrix_row_bytes)));
    op->setAttr(prefix + ".matrix_count", builder.getI32IntegerAttr(static_cast<int32_t>(desc.matrix_count)));
    op->setAttr(prefix + ".alias_of", builder.getI32IntegerAttr(static_cast<int32_t>(desc.alias_of)));
}

bool read_tensor_desc_attrs(mlir::Operation* op,
                            const std::string& prefix,
                            GfxMpsrtTensorAbiDesc& desc) {
    desc = {};
    if (!read_u32_attr(op, prefix + ".rank", desc.rank) ||
        !read_u32_attr(op, prefix + ".dtype", desc.dtype) ||
        !read_u32_attr(op, prefix + ".storage", desc.storage) ||
        !read_u32_attr(op, prefix + ".layout", desc.layout)) {
        return false;
    }
    desc.rank = std::min<uint32_t>(desc.rank, 8u);
    (void)read_u32_attr(op, prefix + ".flags", desc.flags);
    (void)read_u64_attr(op, prefix + ".byte_offset", desc.byte_offset);
    (void)read_u64_attr(op, prefix + ".byte_length", desc.byte_length);
    read_u64_array_attr(op, prefix + ".dims", desc.dims, desc.rank);
    read_i64_array_attr(op, prefix + ".strides", desc.strides, desc.rank);
    (void)read_u32_attr(op, prefix + ".image_width", desc.image_width);
    (void)read_u32_attr(op, prefix + ".image_height", desc.image_height);
    (void)read_u32_attr(op, prefix + ".image_feature_channels", desc.image_feature_channels);
    (void)read_u32_attr(op, prefix + ".image_batch", desc.image_batch);
    (void)read_u32_attr(op, prefix + ".matrix_rows", desc.matrix_rows);
    (void)read_u32_attr(op, prefix + ".matrix_columns", desc.matrix_columns);
    (void)read_u32_attr(op, prefix + ".matrix_row_bytes", desc.matrix_row_bytes);
    (void)read_u32_attr(op, prefix + ".matrix_count", desc.matrix_count);
    (void)read_u32_attr(op, prefix + ".alias_of", desc.alias_of);
    return true;
}

void set_tensor_descs_attrs(mlir::Operation* op,
                            mlir::Builder& builder,
                            const std::vector<GfxMpsrtTensorAbiDesc>& descs) {
    op->setAttr("gfx.mpsrt.runtime_abi.tensor_desc_count",
                builder.getI32IntegerAttr(static_cast<int32_t>(descs.size())));
    for (size_t i = 0; i < descs.size(); ++i) {
        set_tensor_desc_attrs(op,
                              builder,
                              "gfx.mpsrt.runtime_abi.tensor_desc" + std::to_string(i),
                              descs[i]);
    }
}

bool read_tensor_descs_attrs(mlir::Operation* op,
                             std::vector<GfxMpsrtTensorAbiDesc>& descs) {
    descs.clear();
    uint32_t count = 0;
    if (!read_u32_attr(op, "gfx.mpsrt.runtime_abi.tensor_desc_count", count)) {
        return false;
    }
    descs.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        GfxMpsrtTensorAbiDesc desc{};
        if (!read_tensor_desc_attrs(op,
                                    "gfx.mpsrt.runtime_abi.tensor_desc" + std::to_string(i),
                                    desc)) {
            descs.clear();
            return false;
        }
        descs.push_back(desc);
    }
    return true;
}

void set_gemm_desc_attrs(mlir::Operation* op,
                         mlir::Builder& builder,
                         const GfxMpsrtGemmAbiDesc& desc) {
    op->setAttr("gfx.mpsrt.runtime_abi.gemm.transpose_lhs",
                builder.getI32IntegerAttr(static_cast<int32_t>(desc.transpose_lhs)));
    op->setAttr("gfx.mpsrt.runtime_abi.gemm.transpose_rhs",
                builder.getI32IntegerAttr(static_cast<int32_t>(desc.transpose_rhs)));
    op->setAttr("gfx.mpsrt.runtime_abi.gemm.accumulate_fp32",
                builder.getI32IntegerAttr(static_cast<int32_t>(desc.accumulate_fp32)));
    op->setAttr("gfx.mpsrt.runtime_abi.gemm.alpha", builder.getF32FloatAttr(desc.alpha));
    op->setAttr("gfx.mpsrt.runtime_abi.gemm.beta", builder.getF32FloatAttr(desc.beta));
}

void read_gemm_desc_attrs(mlir::Operation* op, GfxMpsrtGemmAbiDesc& desc) {
    (void)read_u32_attr(op, "gfx.mpsrt.runtime_abi.gemm.transpose_lhs", desc.transpose_lhs);
    (void)read_u32_attr(op, "gfx.mpsrt.runtime_abi.gemm.transpose_rhs", desc.transpose_rhs);
    (void)read_u32_attr(op, "gfx.mpsrt.runtime_abi.gemm.accumulate_fp32", desc.accumulate_fp32);
    if (auto attr = op->getAttrOfType<mlir::FloatAttr>("gfx.mpsrt.runtime_abi.gemm.alpha")) {
        desc.alpha = static_cast<float>(attr.getValueAsDouble());
    }
    if (auto attr = op->getAttrOfType<mlir::FloatAttr>("gfx.mpsrt.runtime_abi.gemm.beta")) {
        desc.beta = static_cast<float>(attr.getValueAsDouble());
    }
}

void set_u32_pair_attr(mlir::Operation* op,
                       mlir::Builder& builder,
                       const std::string& name,
                       const uint32_t values[2]) {
    const std::vector<uint32_t> vector_values{values[0], values[1]};
    set_u32_vector_attr(op, builder, name, vector_values);
}

void set_u32_quad_attr(mlir::Operation* op,
                       mlir::Builder& builder,
                       const std::string& name,
                       const uint32_t values[4]) {
    const std::vector<uint32_t> vector_values{values[0], values[1], values[2], values[3]};
    set_u32_vector_attr(op, builder, name, vector_values);
}

void copy_u32_vector_to_fixed(const std::vector<uint32_t>& values, uint32_t* out, size_t count) {
    const size_t n = std::min(count, values.size());
    for (size_t i = 0; i < n; ++i) {
        out[i] = values[i];
    }
}

void set_conv2d_desc_attrs(mlir::Operation* op,
                           mlir::Builder& builder,
                           const GfxMpsrtConv2DAbiDesc& desc) {
    op->setAttr("gfx.mpsrt.runtime_abi.conv2d.groups",
                builder.getI32IntegerAttr(static_cast<int32_t>(desc.groups)));
    set_u32_pair_attr(op, builder, "gfx.mpsrt.runtime_abi.conv2d.strides", desc.strides);
    set_u32_pair_attr(op, builder, "gfx.mpsrt.runtime_abi.conv2d.dilations", desc.dilations);
    set_u32_quad_attr(op, builder, "gfx.mpsrt.runtime_abi.conv2d.pads", desc.pads);
    op->setAttr("gfx.mpsrt.runtime_abi.conv2d.fused_activation",
                builder.getI32IntegerAttr(static_cast<int32_t>(desc.fused_activation)));
    op->setAttr("gfx.mpsrt.runtime_abi.conv2d.accumulate_fp32",
                builder.getI32IntegerAttr(static_cast<int32_t>(desc.accumulate_fp32)));
}

void read_conv2d_desc_attrs(mlir::Operation* op, GfxMpsrtConv2DAbiDesc& desc) {
    (void)read_u32_attr(op, "gfx.mpsrt.runtime_abi.conv2d.groups", desc.groups);
    copy_u32_vector_to_fixed(read_u32_vector_attr(op, "gfx.mpsrt.runtime_abi.conv2d.strides"), desc.strides, 2);
    copy_u32_vector_to_fixed(read_u32_vector_attr(op, "gfx.mpsrt.runtime_abi.conv2d.dilations"), desc.dilations, 2);
    copy_u32_vector_to_fixed(read_u32_vector_attr(op, "gfx.mpsrt.runtime_abi.conv2d.pads"), desc.pads, 4);
    (void)read_u32_attr(op, "gfx.mpsrt.runtime_abi.conv2d.fused_activation", desc.fused_activation);
    (void)read_u32_attr(op, "gfx.mpsrt.runtime_abi.conv2d.accumulate_fp32", desc.accumulate_fp32);
}

void set_pool2d_desc_attrs(mlir::Operation* op,
                           mlir::Builder& builder,
                           const GfxMpsrtPool2DAbiDesc& desc) {
    op->setAttr("gfx.mpsrt.runtime_abi.pool2d.is_avg",
                builder.getI32IntegerAttr(static_cast<int32_t>(desc.is_avg)));
    set_u32_pair_attr(op, builder, "gfx.mpsrt.runtime_abi.pool2d.kernel", desc.kernel);
    set_u32_pair_attr(op, builder, "gfx.mpsrt.runtime_abi.pool2d.strides", desc.strides);
    set_u32_pair_attr(op, builder, "gfx.mpsrt.runtime_abi.pool2d.dilations", desc.dilations);
    set_u32_quad_attr(op, builder, "gfx.mpsrt.runtime_abi.pool2d.pads", desc.pads);
    op->setAttr("gfx.mpsrt.runtime_abi.pool2d.exclude_pad",
                builder.getI32IntegerAttr(static_cast<int32_t>(desc.exclude_pad)));
}

void read_pool2d_desc_attrs(mlir::Operation* op, GfxMpsrtPool2DAbiDesc& desc) {
    (void)read_u32_attr(op, "gfx.mpsrt.runtime_abi.pool2d.is_avg", desc.is_avg);
    copy_u32_vector_to_fixed(read_u32_vector_attr(op, "gfx.mpsrt.runtime_abi.pool2d.kernel"), desc.kernel, 2);
    copy_u32_vector_to_fixed(read_u32_vector_attr(op, "gfx.mpsrt.runtime_abi.pool2d.strides"), desc.strides, 2);
    copy_u32_vector_to_fixed(read_u32_vector_attr(op, "gfx.mpsrt.runtime_abi.pool2d.dilations"), desc.dilations, 2);
    copy_u32_vector_to_fixed(read_u32_vector_attr(op, "gfx.mpsrt.runtime_abi.pool2d.pads"), desc.pads, 4);
    (void)read_u32_attr(op, "gfx.mpsrt.runtime_abi.pool2d.exclude_pad", desc.exclude_pad);
}

void set_resize2d_desc_attrs(mlir::Operation* op,
                             mlir::Builder& builder,
                             const GfxMpsrtResize2DAbiDesc& desc) {
    op->setAttr("gfx.mpsrt.runtime_abi.resize2d.nearest",
                builder.getI32IntegerAttr(static_cast<int32_t>(desc.nearest)));
    op->setAttr("gfx.mpsrt.runtime_abi.resize2d.align_corners",
                builder.getI32IntegerAttr(static_cast<int32_t>(desc.align_corners)));
    op->setAttr("gfx.mpsrt.runtime_abi.resize2d.half_pixel_centers",
                builder.getI32IntegerAttr(static_cast<int32_t>(desc.half_pixel_centers)));
}

void read_resize2d_desc_attrs(mlir::Operation* op, GfxMpsrtResize2DAbiDesc& desc) {
    (void)read_u32_attr(op, "gfx.mpsrt.runtime_abi.resize2d.nearest", desc.nearest);
    (void)read_u32_attr(op, "gfx.mpsrt.runtime_abi.resize2d.align_corners", desc.align_corners);
    (void)read_u32_attr(op, "gfx.mpsrt.runtime_abi.resize2d.half_pixel_centers", desc.half_pixel_centers);
}

void set_softmax_desc_attrs(mlir::Operation* op,
                            mlir::Builder& builder,
                            const GfxMpsrtSoftmaxAbiDesc& desc) {
    op->setAttr("gfx.mpsrt.runtime_abi.softmax.axis",
                builder.getI32IntegerAttr(static_cast<int32_t>(desc.axis)));
    op->setAttr("gfx.mpsrt.runtime_abi.softmax.log_softmax",
                builder.getI32IntegerAttr(static_cast<int32_t>(desc.log_softmax)));
}

void read_softmax_desc_attrs(mlir::Operation* op, GfxMpsrtSoftmaxAbiDesc& desc) {
    (void)read_u32_attr(op, "gfx.mpsrt.runtime_abi.softmax.axis", desc.axis);
    (void)read_u32_attr(op, "gfx.mpsrt.runtime_abi.softmax.log_softmax", desc.log_softmax);
}

void set_topk_desc_attrs(mlir::Operation* op,
                         mlir::Builder& builder,
                         const GfxMpsrtTopKAbiDesc& desc) {
    op->setAttr("gfx.mpsrt.runtime_abi.topk.axis",
                builder.getI32IntegerAttr(static_cast<int32_t>(desc.axis)));
    op->setAttr("gfx.mpsrt.runtime_abi.topk.k",
                builder.getI32IntegerAttr(static_cast<int32_t>(desc.k)));
    op->setAttr("gfx.mpsrt.runtime_abi.topk.mode_max",
                builder.getI32IntegerAttr(static_cast<int32_t>(desc.mode_max)));
    op->setAttr("gfx.mpsrt.runtime_abi.topk.sort_type",
                builder.getI32IntegerAttr(static_cast<int32_t>(desc.sort_type)));
}

void read_topk_desc_attrs(mlir::Operation* op, GfxMpsrtTopKAbiDesc& desc) {
    (void)read_u32_attr(op, "gfx.mpsrt.runtime_abi.topk.axis", desc.axis);
    (void)read_u32_attr(op, "gfx.mpsrt.runtime_abi.topk.k", desc.k);
    (void)read_u32_attr(op, "gfx.mpsrt.runtime_abi.topk.mode_max", desc.mode_max);
    (void)read_u32_attr(op, "gfx.mpsrt.runtime_abi.topk.sort_type", desc.sort_type);
}

void set_msl_dispatch_desc_attrs(mlir::Operation* op,
                                 mlir::Builder& builder,
                                 const GfxMpsrtMslDispatchAbiDesc& desc) {
    op->setAttr("gfx.mpsrt.runtime_abi.msl_dispatch.kernel_family",
                builder.getI32IntegerAttr(static_cast<int32_t>(desc.kernel_family)));
    op->setAttr("gfx.mpsrt.runtime_abi.msl_dispatch.storage",
                builder.getI32IntegerAttr(static_cast<int32_t>(desc.storage)));
    op->setAttr("gfx.mpsrt.runtime_abi.msl_dispatch.layout",
                builder.getI32IntegerAttr(static_cast<int32_t>(desc.layout)));
    op->setAttr("gfx.mpsrt.runtime_abi.msl_dispatch.threads_per_threadgroup",
                builder.getI32IntegerAttr(static_cast<int32_t>(desc.threads_per_threadgroup)));
    op->setAttr("gfx.mpsrt.runtime_abi.msl_dispatch.input_count",
                builder.getI32IntegerAttr(static_cast<int32_t>(desc.input_count)));
    op->setAttr("gfx.mpsrt.runtime_abi.msl_dispatch.output_count",
                builder.getI32IntegerAttr(static_cast<int32_t>(desc.output_count)));
    op->setAttr("gfx.mpsrt.runtime_abi.msl_dispatch.flags",
                builder.getI32IntegerAttr(static_cast<int32_t>(desc.flags)));
}

void read_msl_dispatch_desc_attrs(mlir::Operation* op, GfxMpsrtMslDispatchAbiDesc& desc) {
    (void)read_u32_attr(op, "gfx.mpsrt.runtime_abi.msl_dispatch.kernel_family", desc.kernel_family);
    (void)read_u32_attr(op, "gfx.mpsrt.runtime_abi.msl_dispatch.storage", desc.storage);
    (void)read_u32_attr(op, "gfx.mpsrt.runtime_abi.msl_dispatch.layout", desc.layout);
    (void)read_u32_attr(op,
                        "gfx.mpsrt.runtime_abi.msl_dispatch.threads_per_threadgroup",
                        desc.threads_per_threadgroup);
    (void)read_u32_attr(op, "gfx.mpsrt.runtime_abi.msl_dispatch.input_count", desc.input_count);
    (void)read_u32_attr(op, "gfx.mpsrt.runtime_abi.msl_dispatch.output_count", desc.output_count);
    (void)read_u32_attr(op, "gfx.mpsrt.runtime_abi.msl_dispatch.flags", desc.flags);
}

void annotate_runtime_abi_stage_desc(mlir::Operation* op,
                                     mlir::Builder& builder,
                                     const GfxMpsrtBuilderRecord& record) {
    set_tensor_descs_attrs(op, builder, record.tensor_descs);
    if (!record.dispatch_kernel_family.empty()) {
        op->setAttr("gfx.mpsrt.runtime_abi.dispatch_kernel_family",
                    builder.getStringAttr(record.dispatch_kernel_family));
    }
    if (!record.dispatch_entry_point.empty()) {
        op->setAttr("gfx.mpsrt.runtime_abi.dispatch_entry_point",
                    builder.getStringAttr(record.dispatch_entry_point));
    }
    if (record.dispatch_kernel_family_id != 0) {
        op->setAttr("gfx.mpsrt.runtime_abi.dispatch_kernel_family_id",
                    builder.getI32IntegerAttr(static_cast<int32_t>(record.dispatch_kernel_family_id)));
    }
    if (record.dispatch_flags != GfxMpsrtMslDispatchFlagNone) {
        op->setAttr("gfx.mpsrt.runtime_abi.dispatch_flags",
                    builder.getI32IntegerAttr(static_cast<int32_t>(record.dispatch_flags)));
    }
    if (record.dispatch_threads_per_threadgroup != 0) {
        op->setAttr("gfx.mpsrt.runtime_abi.dispatch_threads_per_threadgroup",
                    builder.getI32IntegerAttr(static_cast<int32_t>(record.dispatch_threads_per_threadgroup)));
    }
    if (record.dispatch_precompiled_kernel_required) {
        op->setAttr("gfx.mpsrt.runtime_abi.dispatch_precompiled_kernel_required", builder.getBoolAttr(true));
    }

    if (record.stage_kind == GfxMpsrtStageKind::MSLDispatch) {
        set_msl_dispatch_desc_attrs(op, builder, record.msl_dispatch_desc);
    } else if (gfx_mpsrt_stage_uses_conv2d_desc(record.stage_kind)) {
        set_conv2d_desc_attrs(op, builder, record.conv2d_desc);
    } else if (record.stage_kind == GfxMpsrtStageKind::MPSGemm) {
        set_gemm_desc_attrs(op, builder, record.gemm_desc);
    } else if (gfx_mpsrt_stage_uses_pool2d_desc(record.stage_kind)) {
        set_pool2d_desc_attrs(op, builder, record.pool2d_desc);
    } else if (gfx_mpsrt_stage_uses_resize2d_desc(record.stage_kind)) {
        set_resize2d_desc_attrs(op, builder, record.resize2d_desc);
    } else if (gfx_mpsrt_stage_uses_softmax_desc(record.stage_kind)) {
        set_softmax_desc_attrs(op, builder, record.softmax_desc);
    } else if (gfx_mpsrt_stage_uses_topk_desc(record.stage_kind)) {
        set_topk_desc_attrs(op, builder, record.topk_desc);
    }
}

void read_runtime_abi_stage_desc(mlir::Operation* op, GfxMpsrtBuilderRecord& record) {
    (void)read_tensor_descs_attrs(op, record.tensor_descs);
    if (auto family = op->getAttrOfType<mlir::StringAttr>("gfx.mpsrt.runtime_abi.dispatch_kernel_family")) {
        record.dispatch_kernel_family = family.str();
    }
    if (auto entry = op->getAttrOfType<mlir::StringAttr>("gfx.mpsrt.runtime_abi.dispatch_entry_point")) {
        record.dispatch_entry_point = entry.str();
    }
    (void)read_u32_attr(op,
                        "gfx.mpsrt.runtime_abi.dispatch_kernel_family_id",
                        record.dispatch_kernel_family_id);
    (void)read_u32_attr(op, "gfx.mpsrt.runtime_abi.dispatch_flags", record.dispatch_flags);
    (void)read_u32_attr(op,
                        "gfx.mpsrt.runtime_abi.dispatch_threads_per_threadgroup",
                        record.dispatch_threads_per_threadgroup);
    (void)read_bool_attr(op,
                         "gfx.mpsrt.runtime_abi.dispatch_precompiled_kernel_required",
                         record.dispatch_precompiled_kernel_required);

    if (record.stage_kind == GfxMpsrtStageKind::MSLDispatch) {
        read_msl_dispatch_desc_attrs(op, record.msl_dispatch_desc);
    } else if (gfx_mpsrt_stage_uses_conv2d_desc(record.stage_kind)) {
        read_conv2d_desc_attrs(op, record.conv2d_desc);
    } else if (record.stage_kind == GfxMpsrtStageKind::MPSGemm) {
        read_gemm_desc_attrs(op, record.gemm_desc);
    } else if (gfx_mpsrt_stage_uses_pool2d_desc(record.stage_kind)) {
        read_pool2d_desc_attrs(op, record.pool2d_desc);
    } else if (gfx_mpsrt_stage_uses_resize2d_desc(record.stage_kind)) {
        read_resize2d_desc_attrs(op, record.resize2d_desc);
    } else if (gfx_mpsrt_stage_uses_softmax_desc(record.stage_kind)) {
        read_softmax_desc_attrs(op, record.softmax_desc);
    } else if (gfx_mpsrt_stage_uses_topk_desc(record.stage_kind)) {
        read_topk_desc_attrs(op, record.topk_desc);
    }
}

void set_storage_bridge_attrs(mlir::Operation* op,
                              mlir::Builder& builder,
                              const std::string& prefix,
                              const GfxMpsrtStorageBridgeDesc& bridge) {
    op->setAttr(prefix + ".value", builder.getI32IntegerAttr(static_cast<int32_t>(bridge.value)));
    op->setAttr(prefix + ".direction", builder.getI32IntegerAttr(static_cast<int32_t>(bridge.direction)));
    op->setAttr(prefix + ".source_storage",
                builder.getI32IntegerAttr(static_cast<int32_t>(bridge.source_storage)));
    op->setAttr(prefix + ".target_storage",
                builder.getI32IntegerAttr(static_cast<int32_t>(bridge.target_storage)));
    set_tensor_desc_attrs(op, builder, prefix + ".tensor", bridge.tensor);
}

bool read_storage_bridge_attrs(mlir::Operation* op,
                               const std::string& prefix,
                               GfxMpsrtStorageBridgeDesc& bridge) {
    uint32_t direction = 0;
    uint32_t source_storage = 0;
    uint32_t target_storage = 0;
    if (!read_u32_attr(op, prefix + ".value", bridge.value) ||
        !read_u32_attr(op, prefix + ".direction", direction) ||
        !read_u32_attr(op, prefix + ".source_storage", source_storage) ||
        !read_u32_attr(op, prefix + ".target_storage", target_storage) ||
        !read_tensor_desc_attrs(op, prefix + ".tensor", bridge.tensor)) {
        return false;
    }
    bridge.direction = static_cast<GfxMpsrtStorageBridgeDirection>(direction);
    bridge.source_storage = static_cast<GfxMpsrtStorage>(source_storage);
    bridge.target_storage = static_cast<GfxMpsrtStorage>(target_storage);
    GfxMpsrtStorageBridgeDesc normalized{};
    if (!gfx_mpsrt_make_storage_bridge_desc(bridge.value,
                                            bridge.tensor,
                                            bridge.direction,
                                            normalized)) {
        return false;
    }
    return normalized.source_storage == bridge.source_storage &&
           normalized.target_storage == bridge.target_storage;
}

void annotate_runtime_abi_storage_bridges(mlir::func::FuncOp plan_func,
                                          mlir::Builder& builder,
                                          const std::vector<GfxMpsrtStorageBridgeDesc>& bridges) {
    plan_func->setAttr("gfx.mpsrt.runtime_abi.storage_bridge_count",
                       builder.getI32IntegerAttr(static_cast<int32_t>(bridges.size())));
    for (size_t i = 0; i < bridges.size(); ++i) {
        set_storage_bridge_attrs(plan_func,
                                 builder,
                                 "gfx.mpsrt.runtime_abi.storage_bridge" + std::to_string(i),
                                 bridges[i]);
    }
}

bool read_runtime_abi_storage_bridges(mlir::func::FuncOp plan_func,
                                      std::vector<GfxMpsrtStorageBridgeDesc>& bridges) {
    bridges.clear();
    uint32_t bridge_count = 0;
    if (!read_u32_attr(plan_func, "gfx.mpsrt.runtime_abi.storage_bridge_count", bridge_count)) {
        return true;
    }
    bridges.reserve(bridge_count);
    for (uint32_t i = 0; i < bridge_count; ++i) {
        GfxMpsrtStorageBridgeDesc bridge{};
        if (!read_storage_bridge_attrs(plan_func,
                                       "gfx.mpsrt.runtime_abi.storage_bridge" + std::to_string(i),
                                       bridge)) {
            bridges.clear();
            return false;
        }
        bridges.push_back(bridge);
    }
    return true;
}

void annotate_runtime_abi_call(mlir::func::CallOp call,
                               mlir::Builder& builder,
                               const GfxMpsrtBuilderRecord& record,
                               size_t record_index) {
    call->setAttr("gfx.mpsrt.runtime_abi.record_index",
                  builder.getI32IntegerAttr(static_cast<int32_t>(record_index)));
    call->setAttr("gfx.mpsrt.runtime_abi.record_kind",
                  builder.getStringAttr(gfx_mpsrt_builder_record_kind_name(record.kind)));
    if (!record.stage_record_key.empty()) {
        call->setAttr("gfx.mpsrt.runtime_abi.stage_record_key",
                      builder.getStringAttr(record.stage_record_key));
    }
    if (record.stage_kind != GfxMpsrtStageKind::Unknown) {
        call->setAttr("gfx.mpsrt.runtime_abi.stage_kind",
                      builder.getStringAttr(gfx_mpsrt_stage_kind_name(record.stage_kind)));
    }
    if (!record.kernel_name.empty()) {
        call->setAttr("gfx.mpsrt.runtime_abi.kernel_name", builder.getStringAttr(record.kernel_name));
    }
    if (record.kind == GfxMpsrtBuilderRecordKind::AddTensor) {
        call->setAttr("gfx.mpsrt.runtime_abi.value",
                      builder.getI32IntegerAttr(static_cast<int32_t>(record.value)));
    }
    annotate_runtime_abi_stage_desc(call, builder, record);
    set_u32_vector_attr(call, builder, "gfx.mpsrt.runtime_abi.input_values", record.inputs);
    set_u32_vector_attr(call, builder, "gfx.mpsrt.runtime_abi.output_values", record.outputs);
    set_u32_vector_attr(call, builder, "gfx.mpsrt.runtime_abi.kernel_buffer_order", record.kernel_buffer_order);
}

bool read_runtime_abi_program_plan(mlir::ModuleOp module,
                                   GfxMpsrtRuntimeAbiProgramPlan& out) {
    out = {};
    if (!module) {
        return false;
    }

    GfxMpsrtProgram program{};
    if (!read_module_mpsrt_ops_program(module, program)) {
        return false;
    }
    GfxMpsrtBuilderPlan builder_plan{};
    if (!gfx_mpsrt_build_builder_plan_from_program(program, builder_plan)) {
        return false;
    }

    out.valid = true;
    out.program = std::move(program);
    out.builder_plan = std::move(builder_plan);
    return true;
}

void annotate_runtime_abi_plan_func(mlir::func::FuncOp plan_func,
                                    mlir::Builder& builder,
                                    const GfxMpsrtRuntimeAbiProgramPlan& program_plan) {
    const auto& builder_plan = program_plan.builder_plan;
    plan_func->setAttr("gfx.mpsrt.runtime_abi.generated", builder.getBoolAttr(true));
    plan_func->setAttr("gfx.mpsrt.runtime_abi.kind",
                       builder.getStringAttr(program_plan.program.multi_stage ? "multi_stage" : "single_stage"));
    plan_func->setAttr("gfx.mpsrt.runtime_abi.record_key",
                       builder.getStringAttr(builder_plan.stage_record_key));
    plan_func->setAttr("gfx.mpsrt.runtime_abi.record_count",
                       builder.getI32IntegerAttr(static_cast<int32_t>(builder_plan.records.size())));
    set_u32_vector_attr(plan_func, builder, "gfx.mpsrt.runtime_abi.input_values", builder_plan.input_values);
    set_u32_vector_attr(plan_func, builder, "gfx.mpsrt.runtime_abi.output_values", builder_plan.output_values);
    if (builder_plan.external_buffer_abi_valid) {
        plan_func->setAttr("gfx.mpsrt.runtime_abi.external_buffer_count",
                           builder.getI32IntegerAttr(static_cast<int32_t>(builder_plan.external_buffer_count)));
        plan_func->setAttr("gfx.mpsrt.runtime_abi.external_output_buffer_count",
                           builder.getI32IntegerAttr(
                               static_cast<int32_t>(builder_plan.external_output_buffer_count)));
        std::vector<uint32_t> roles;
        roles.reserve(builder_plan.external_buffer_roles.size());
        for (const auto role : builder_plan.external_buffer_roles) {
            roles.push_back(static_cast<uint32_t>(role));
        }
        set_u32_vector_attr(plan_func, builder, "gfx.mpsrt.runtime_abi.external_buffer_roles", roles);
    }
    annotate_runtime_abi_storage_bridges(plan_func, builder, builder_plan.storage_bridges);
}

bool materialize_runtime_abi_call_plan(mlir::ModuleOp module,
                                       const GfxMpsrtRuntimeAbiProgramPlan& program_plan) {
    if (!module || !program_plan.valid || !program_plan.builder_plan.valid) {
        return false;
    }

    ensure_runtime_abi_dialects(module);
    mlir::OpBuilder builder(module.getContext());
    const auto loc = mlir::UnknownLoc::get(module.getContext());
    const auto empty_func_type = builder.getFunctionType({}, {});

    if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(kRuntimeAbiPlanSymbol)) {
        existing.erase();
    }

    builder.setInsertionPointToEnd(module.getBody());
    std::set<std::string> declared_symbols;
    for (const auto& record : program_plan.builder_plan.records) {
        if (record.symbol.empty() || record.symbol == kRuntimeAbiPlanSymbol ||
            !declared_symbols.insert(record.symbol).second ||
            module.lookupSymbol<mlir::func::FuncOp>(record.symbol)) {
            continue;
        }
        auto decl = mlir::func::FuncOp::create(builder, loc, record.symbol, empty_func_type);
        decl.setSymVisibility("private");
    }

    auto plan_func = mlir::func::FuncOp::create(builder, loc, kRuntimeAbiPlanSymbol, empty_func_type);
    plan_func.setSymVisibility("private");
    annotate_runtime_abi_plan_func(plan_func, builder, program_plan);
    plan_func.addEntryBlock();

    mlir::OpBuilder body_builder(plan_func.getBody());
    body_builder.setInsertionPointToStart(&plan_func.getBody().front());
    for (size_t i = 0; i < program_plan.builder_plan.records.size(); ++i) {
        const auto& record = program_plan.builder_plan.records[i];
        if (record.symbol.empty()) {
            return false;
        }
        auto call = mlir::func::CallOp::create(body_builder,
                                               loc,
                                               record.symbol,
                                               mlir::TypeRange{},
                                               mlir::ValueRange{});
        annotate_runtime_abi_call(call, body_builder, record, i);
    }
    mlir::func::ReturnOp::create(body_builder, loc);

    module->setAttr("gfx.mpsrt.runtime_abi.call_plan_symbol",
                    builder.getStringAttr(kRuntimeAbiPlanSymbol));
    return true;
}

bool mark_runtime_abi_call_plan_materialized(mlir::ModuleOp module, bool materialized) {
    if (!module || !materialized) {
        return false;
    }
    module->removeAttr("gfx.apple.pipeline.runtime_abi.call_plan_deferred");
    module->setAttr("gfx.apple.pipeline.runtime_abi.call_plan_materialized",
                    mlir::BoolAttr::get(module.getContext(), true));
    return true;
}

void erase_runtime_abi_call_plan(mlir::ModuleOp module) {
    if (!module) {
        return;
    }
    if (auto plan_func = module.lookupSymbol<mlir::func::FuncOp>(kRuntimeAbiPlanSymbol)) {
        plan_func.erase();
    }
    module->removeAttr("gfx.mpsrt.runtime_abi.call_plan_symbol");
    module->removeAttr("gfx.apple.pipeline.runtime_abi.call_plan_materialized");
}

class ConvertMpsrtToRuntimeAbiAttrsPass final
    : public mlir::PassWrapper<ConvertMpsrtToRuntimeAbiAttrsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertMpsrtToRuntimeAbiAttrsPass)

    llvm::StringRef getArgument() const final {
        return "gfx-apple-runtime-abi-call-plan";
    }

    llvm::StringRef getDescription() const final {
        return "Materialize the Apple MPSRT host-builder call plan from typed MPSRT ops";
    }

    void runOnOperation() final {
        auto module = getOperation();
        ensure_runtime_abi_dialects(module);

        GfxMpsrtRuntimeAbiProgramPlan program_plan{};
        if (!read_runtime_abi_program_plan(module, program_plan)) {
            return;
        }
        if (!materialize_runtime_abi_call_plan(module, program_plan)) {
            module.emitError("failed to materialize MPSRT runtime ABI call plan from typed MPSRT ops");
            signalPassFailure();
            return;
        }
        erase_module_mpsrt_legacy_attrs(module);
        (void)mark_runtime_abi_call_plan_materialized(module, true);
    }
};

}  // namespace

void populate_gfx_apple_mpsrt_runtime_abi_pipeline(mlir::PassManager& pm) {
    pm.addPass(std::make_unique<ConvertMpsrtToRuntimeAbiAttrsPass>());
}

std::unique_ptr<mlir::Pass> createGfxAppleMpsrtRuntimeAbiCallPlanPass() {
    return std::make_unique<ConvertMpsrtToRuntimeAbiAttrsPass>();
}

bool has_gfx_apple_mpsrt_runtime_abi_call_plan(mlir::ModuleOp module) {
    if (!module) {
        return false;
    }
    ensure_runtime_abi_dialects(module);
    std::string plan_symbol = kRuntimeAbiPlanSymbol;
    if (auto attr = module->getAttrOfType<mlir::StringAttr>("gfx.mpsrt.runtime_abi.call_plan_symbol")) {
        plan_symbol = attr.str();
    }
    auto plan_func = module.lookupSymbol<mlir::func::FuncOp>(plan_symbol);
    return plan_func &&
           static_cast<bool>(plan_func->getAttrOfType<mlir::BoolAttr>("gfx.mpsrt.runtime_abi.generated"));
}

bool materialize_gfx_apple_mpsrt_runtime_abi_call_plan(mlir::ModuleOp module) {
    if (!module) {
        return false;
    }
    GfxMpsrtProgram program{};
    if (!read_module_mpsrt_ops_program(module, program)) {
        return false;
    }
    if (has_gfx_apple_mpsrt_runtime_abi_call_plan(module)) {
        return mark_runtime_abi_call_plan_materialized(module, true);
    }

    mlir::PassManager pm(module.getContext());
    populate_gfx_apple_mpsrt_runtime_abi_pipeline(pm);
    if (mlir::failed(pm.run(module))) {
        return false;
    }
    return mark_runtime_abi_call_plan_materialized(
        module,
        has_gfx_apple_mpsrt_runtime_abi_call_plan(module));
}

bool rematerialize_gfx_apple_mpsrt_runtime_abi_call_plan(mlir::ModuleOp module) {
    if (!module) {
        return false;
    }
    erase_runtime_abi_call_plan(module);
    return materialize_gfx_apple_mpsrt_runtime_abi_call_plan(module);
}

bool read_gfx_apple_mpsrt_runtime_abi_call_plan(mlir::ModuleOp module,
                                                GfxMpsrtBuilderPlan& out) {
    out = {};
    if (!module) {
        return false;
    }
    ensure_runtime_abi_dialects(module);
    std::string plan_symbol = kRuntimeAbiPlanSymbol;
    if (auto attr = module->getAttrOfType<mlir::StringAttr>("gfx.mpsrt.runtime_abi.call_plan_symbol")) {
        plan_symbol = attr.str();
    }
    auto plan_func = module.lookupSymbol<mlir::func::FuncOp>(plan_symbol);
    if (!plan_func ||
        !plan_func->getAttrOfType<mlir::BoolAttr>("gfx.mpsrt.runtime_abi.generated")) {
        return false;
    }

    auto record_key_attr = plan_func->getAttrOfType<mlir::StringAttr>("gfx.mpsrt.runtime_abi.record_key");
    if (!record_key_attr) {
        return false;
    }
    out.stage_record_key = record_key_attr.str();
    out.input_values = read_u32_vector_attr(plan_func, "gfx.mpsrt.runtime_abi.input_values");
    out.output_values = read_u32_vector_attr(plan_func, "gfx.mpsrt.runtime_abi.output_values");

    if (auto external_count =
            plan_func->getAttrOfType<mlir::IntegerAttr>("gfx.mpsrt.runtime_abi.external_buffer_count")) {
        out.external_buffer_abi_valid = true;
        out.external_buffer_count = static_cast<uint32_t>(external_count.getInt());
        if (auto output_count =
                plan_func->getAttrOfType<mlir::IntegerAttr>(
                    "gfx.mpsrt.runtime_abi.external_output_buffer_count")) {
            out.external_output_buffer_count = static_cast<uint32_t>(output_count.getInt());
        }
        const auto role_values =
            read_u32_vector_attr(plan_func, "gfx.mpsrt.runtime_abi.external_buffer_roles");
        out.external_buffer_roles.reserve(role_values.size());
        for (const auto role_value : role_values) {
            out.external_buffer_roles.push_back(static_cast<GfxMpsrtExternalBufferRole>(role_value));
        }
    }

    plan_func.walk([&](mlir::func::CallOp call) {
        GfxMpsrtBuilderRecord record{};
        record.symbol = call.getCallee().str();
        if (auto kind = call->getAttrOfType<mlir::StringAttr>("gfx.mpsrt.runtime_abi.record_kind")) {
            record.kind = builder_record_kind_from_name(kind);
        }
        if (auto key = call->getAttrOfType<mlir::StringAttr>("gfx.mpsrt.runtime_abi.stage_record_key")) {
            record.stage_record_key = key.str();
        }
        if (auto stage_kind = call->getAttrOfType<mlir::StringAttr>("gfx.mpsrt.runtime_abi.stage_kind")) {
            record.stage_kind = gfx_mpsrt_stage_kind_from_name(stage_kind.str());
        }
        if (auto kernel_name = call->getAttrOfType<mlir::StringAttr>("gfx.mpsrt.runtime_abi.kernel_name")) {
            record.kernel_name = kernel_name.str();
        }
        if (auto value = call->getAttrOfType<mlir::IntegerAttr>("gfx.mpsrt.runtime_abi.value")) {
            record.value = static_cast<GfxMpsrtValue>(value.getInt());
        }
        record.inputs = read_u32_vector_attr(call, "gfx.mpsrt.runtime_abi.input_values");
        record.outputs = read_u32_vector_attr(call, "gfx.mpsrt.runtime_abi.output_values");
        record.kernel_buffer_order =
            read_u32_vector_attr(call, "gfx.mpsrt.runtime_abi.kernel_buffer_order");
        read_runtime_abi_stage_desc(call, record);
        out.records.push_back(std::move(record));
    });

    const auto record_count_attr =
        plan_func->getAttrOfType<mlir::IntegerAttr>("gfx.mpsrt.runtime_abi.record_count");
    if (!record_count_attr ||
        static_cast<size_t>(record_count_attr.getInt()) != out.records.size() ||
        out.records.empty() ||
        out.records.front().kind != GfxMpsrtBuilderRecordKind::ModelBegin ||
        out.records.back().kind != GfxMpsrtBuilderRecordKind::ModelEnd) {
        out = {};
        return false;
    }
    if (!read_runtime_abi_storage_bridges(plan_func, out.storage_bridges)) {
        out = {};
        return false;
    }
    if (out.input_values.empty()) {
        for (const auto& record : out.records) {
            if (record.kind == GfxMpsrtBuilderRecordKind::AddTensor) {
                out.input_values.push_back(record.value);
            }
        }
    }
    if (out.output_values.empty()) {
        for (auto it = out.records.rbegin(); it != out.records.rend(); ++it) {
            if (it->kind == GfxMpsrtBuilderRecordKind::EncodeStage) {
                out.output_values = it->outputs;
                break;
            }
        }
    }
    out.valid = !out.stage_record_key.empty() &&
                !out.input_values.empty() &&
                !out.output_values.empty() &&
                std::all_of(out.records.begin(), out.records.end(), [](const auto& record) {
                    if (record.kind == GfxMpsrtBuilderRecordKind::Unknown || record.symbol.empty()) {
                        return false;
                    }
                    if (record.kind == GfxMpsrtBuilderRecordKind::AddTensor) {
                        return record.tensor_descs.size() == 1;
                    }
                    if (record.kind == GfxMpsrtBuilderRecordKind::EncodeStage) {
                        return record.stage_kind != GfxMpsrtStageKind::Unknown &&
                               !record.outputs.empty() &&
                               record.tensor_descs.size() == record.outputs.size();
                    }
                    return true;
                });
    if (!out.valid) {
        out = {};
    }
    return out.valid;
}

}  // namespace gfx_plugin
}  // namespace ov
