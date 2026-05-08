// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/gfx_mpsrt_dialect.hpp"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringRef.h"

namespace ov {
namespace gfx_plugin {
namespace mpsrt {
namespace {

llvm::StringRef expected_stage_kind_for_op(llvm::StringRef op_name) {
    if (op_name == "gfx.mpsrt.conv2d") {
        return "mps_conv2d";
    }
    if (op_name == "gfx.mpsrt.group_conv2d") {
        return "mps_group_conv2d";
    }
    if (op_name == "gfx.mpsrt.pool2d") {
        return "mps_pool2d";
    }
    if (op_name == "gfx.mpsrt.resize2d") {
        return "mps_resize2d";
    }
    if (op_name == "gfx.mpsrt.gemm") {
        return "mps_gemm";
    }
    if (op_name == "gfx.mpsrt.softmax") {
        return "mps_softmax";
    }
    if (op_name == "gfx.mpsrt.topk") {
        return "mps_topk";
    }
    if (op_name == "gfx.mpsrt.dispatch") {
        return "msl_dispatch";
    }
    return {};
}

llvm::StringRef expected_target_storage_for_conversion_op(llvm::StringRef op_name) {
    if (op_name == "gfx.mpsrt.to_image") {
        return "image";
    }
    if (op_name == "gfx.mpsrt.to_matrix") {
        return "matrix";
    }
    if (op_name == "gfx.mpsrt.to_ndarray") {
        return "ndarray";
    }
    if (op_name == "gfx.mpsrt.to_buffer") {
        return "buffer";
    }
    if (op_name == "gfx.mpsrt.alias") {
        return "alias";
    }
    return {};
}

bool has_string_attr(mlir::Operation* op, llvm::StringRef name) {
    return static_cast<bool>(op->getAttrOfType<mlir::StringAttr>(name));
}

bool has_integer_attr(mlir::Operation* op, llvm::StringRef name) {
    return static_cast<bool>(op->getAttrOfType<mlir::IntegerAttr>(name));
}

bool has_array_attr(mlir::Operation* op, llvm::StringRef name) {
    return static_cast<bool>(op->getAttrOfType<mlir::ArrayAttr>(name));
}

mlir::LogicalResult require_string_attr(mlir::Operation* op, llvm::StringRef name) {
    if (has_string_attr(op, name)) {
        return mlir::success();
    }
    return op->emitOpError() << "requires string attribute '" << name << "'";
}

mlir::LogicalResult require_integer_attr(mlir::Operation* op, llvm::StringRef name) {
    if (has_integer_attr(op, name)) {
        return mlir::success();
    }
    return op->emitOpError() << "requires integer attribute '" << name << "'";
}

mlir::LogicalResult require_array_attr(mlir::Operation* op, llvm::StringRef name) {
    if (has_array_attr(op, name)) {
        return mlir::success();
    }
    return op->emitOpError() << "requires array attribute '" << name << "'";
}

mlir::LogicalResult verify_storage_conversion_op(mlir::Operation* op,
                                                 llvm::StringRef op_name,
                                                 llvm::StringRef expected_target_storage) {
    if (mlir::failed(require_integer_attr(op, "gfx.mpsrt.storage_bridge.value")) ||
        mlir::failed(require_string_attr(op, "gfx.mpsrt.storage_bridge.direction")) ||
        mlir::failed(require_string_attr(op, "gfx.mpsrt.storage_bridge.source_storage")) ||
        mlir::failed(require_string_attr(op, "gfx.mpsrt.storage_bridge.target_storage")) ||
        mlir::failed(require_string_attr(op, "gfx.mpsrt.storage_bridge.tensor.storage"))) {
        return mlir::failure();
    }
    auto target_storage = op->getAttrOfType<mlir::StringAttr>(
        "gfx.mpsrt.storage_bridge.target_storage");
    if (target_storage.strref() != expected_target_storage) {
        return op->emitOpError() << "target storage '" << target_storage.strref()
                                 << "' does not match op name '" << op_name
                                 << "'; expected '" << expected_target_storage << "'";
    }
    return mlir::success();
}

}  // namespace

mlir::LogicalResult verify_gfx_mpsrt_op(mlir::Operation* op,
                                        llvm::StringRef op_name) {
    if (!op) {
        return mlir::failure();
    }
    if (op_name == "gfx.mpsrt.return") {
        return require_array_attr(op, "gfx.mpsrt.op.output_values");
    }
    if (const auto expected_target_storage = expected_target_storage_for_conversion_op(op_name);
        !expected_target_storage.empty()) {
        return verify_storage_conversion_op(op, op_name, expected_target_storage);
    }

    const auto expected_stage_kind = expected_stage_kind_for_op(op_name);
    if (expected_stage_kind.empty()) {
        return op->emitOpError() << "has unknown MPSRT op name '" << op_name << "'";
    }
    if (mlir::failed(require_integer_attr(op, "gfx.mpsrt.op.stage_index")) ||
        mlir::failed(require_array_attr(op, "gfx.mpsrt.op.input_values")) ||
        mlir::failed(require_array_attr(op, "gfx.mpsrt.op.output_values")) ||
        mlir::failed(require_integer_attr(op, "gfx.mpsrt.op.output_count")) ||
        mlir::failed(require_string_attr(op, "gfx.mpsrt.op.stage.backend")) ||
        mlir::failed(require_string_attr(op, "gfx.mpsrt.op.stage.stage_kind")) ||
        mlir::failed(require_string_attr(op, "gfx.mpsrt.op.stage.stage_record_key")) ||
        mlir::failed(require_string_attr(op, "gfx.stage_manifest.stage_family")) ||
        mlir::failed(require_string_attr(op, "gfx.stage_manifest.backend_domain")) ||
        mlir::failed(require_string_attr(op, "gfx.stage_manifest.execution_kind")) ||
        mlir::failed(require_string_attr(op, "gfx.stage_manifest.storage"))) {
        return mlir::failure();
    }

    auto stage_kind = op->getAttrOfType<mlir::StringAttr>("gfx.mpsrt.op.stage.stage_kind");
    if (stage_kind.strref() != expected_stage_kind) {
        return op->emitOpError() << "stage kind '" << stage_kind.strref()
                                 << "' does not match op name; expected '"
                                 << expected_stage_kind << "'";
    }
    if (op_name == "gfx.mpsrt.dispatch" &&
        mlir::failed(require_string_attr(op, "gfx.stage_manifest.kernel.entry_point"))) {
        return mlir::failure();
    }
    return mlir::success();
}

GfxMpsrtDialect::GfxMpsrtDialect(mlir::MLIRContext* context)
    : mlir::Dialect(getDialectNamespace(), context, mlir::TypeID::get<GfxMpsrtDialect>()) {
    addOperations<Conv2DOp,
                  GroupConv2DOp,
                  Pool2DOp,
                  Resize2DOp,
                  GemmOp,
                  SoftmaxOp,
                  TopKOp,
                  ToImageOp,
                  ToMatrixOp,
                  ToNDArrayOp,
                  ToBufferOp,
                  AliasOp,
                  DispatchOp,
                  ReturnOp>();
}

void register_gfx_mpsrt_dialect(mlir::MLIRContext& context) {
    context.loadDialect<GfxMpsrtDialect>();
}

}  // namespace mpsrt
}  // namespace gfx_plugin
}  // namespace ov
