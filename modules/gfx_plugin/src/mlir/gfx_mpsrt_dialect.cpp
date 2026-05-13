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

struct ExpectedStageManifest {
    llvm::StringRef backend_domain;
    llvm::StringRef execution_kind;
    llvm::StringRef stage_family;
};

ExpectedStageManifest expected_manifest_for_stage_op(llvm::StringRef op_name) {
    if (op_name == "gfx.mpsrt.conv2d") {
        return {"apple_mps", "vendor_primitive", "convolution"};
    }
    if (op_name == "gfx.mpsrt.group_conv2d") {
        return {"apple_mps", "vendor_primitive", "group_convolution"};
    }
    if (op_name == "gfx.mpsrt.pool2d") {
        return {"apple_mps", "vendor_primitive", "pooling"};
    }
    if (op_name == "gfx.mpsrt.resize2d") {
        return {"apple_mps", "vendor_primitive", "resize"};
    }
    if (op_name == "gfx.mpsrt.gemm") {
        return {"apple_mps", "vendor_primitive", "gemm"};
    }
    if (op_name == "gfx.mpsrt.softmax") {
        return {"apple_mps", "vendor_primitive", "softmax"};
    }
    if (op_name == "gfx.mpsrt.topk") {
        return {"apple_mps", "vendor_primitive", "topk"};
    }
    if (op_name == "gfx.mpsrt.dispatch") {
        return {"apple_msl", "custom_kernel", {}};
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

    const auto expected_manifest = expected_manifest_for_stage_op(op_name);
    if (expected_manifest.backend_domain.empty()) {
        return op->emitOpError() << "has unknown MPSRT op name '" << op_name << "'";
    }
    if (mlir::failed(require_integer_attr(op, "gfx.mpsrt.op.stage_index")) ||
        mlir::failed(require_array_attr(op, "gfx.mpsrt.op.input_values")) ||
        mlir::failed(require_array_attr(op, "gfx.mpsrt.op.output_values")) ||
        mlir::failed(require_integer_attr(op, "gfx.mpsrt.op.output_count")) ||
        mlir::failed(require_string_attr(op, "gfx.stage_manifest.stage_family")) ||
        mlir::failed(require_string_attr(op, "gfx.stage_manifest.backend_domain")) ||
        mlir::failed(require_string_attr(op, "gfx.stage_manifest.execution_kind")) ||
        mlir::failed(require_string_attr(op, "gfx.stage_manifest.storage"))) {
        return mlir::failure();
    }

    auto backend_domain = op->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.backend_domain");
    if (backend_domain.strref() != expected_manifest.backend_domain) {
        return op->emitOpError() << "manifest backend domain '" << backend_domain.strref()
                                 << "' does not match op name; expected '"
                                 << expected_manifest.backend_domain << "'";
    }
    auto execution_kind = op->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.execution_kind");
    if (execution_kind.strref() != expected_manifest.execution_kind) {
        return op->emitOpError() << "manifest execution kind '" << execution_kind.strref()
                                 << "' does not match op name; expected '"
                                 << expected_manifest.execution_kind << "'";
    }
    if (!expected_manifest.stage_family.empty()) {
        auto stage_family = op->getAttrOfType<mlir::StringAttr>("gfx.stage_manifest.stage_family");
        if (stage_family.strref() != expected_manifest.stage_family) {
            return op->emitOpError() << "manifest stage family '" << stage_family.strref()
                                     << "' does not match op name; expected '"
                                     << expected_manifest.stage_family << "'";
        }
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
