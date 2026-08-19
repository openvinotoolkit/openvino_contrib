// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace ov {
namespace gfx_plugin {
namespace mpsrt {

mlir::LogicalResult verify_gfx_mpsrt_op(mlir::Operation* op,
                                        llvm::StringRef op_name);

class GfxMpsrtDialect final : public mlir::Dialect {
public:
    explicit GfxMpsrtDialect(mlir::MLIRContext* context);

    static llvm::StringRef getDialectNamespace() {
        return "gfx";
    }
};

template <typename ConcreteType>
class GfxMpsrtOp : public mlir::Op<ConcreteType,
                                   mlir::OpTrait::ZeroOperands,
                                   mlir::OpTrait::ZeroResults,
                                   mlir::OpTrait::ZeroRegions,
                                   mlir::OpTrait::ZeroSuccessors> {
public:
    using mlir::Op<ConcreteType,
                   mlir::OpTrait::ZeroOperands,
                   mlir::OpTrait::ZeroResults,
                   mlir::OpTrait::ZeroRegions,
                   mlir::OpTrait::ZeroSuccessors>::Op;

    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() {
        return {};
    }

    mlir::LogicalResult verify() {
        return verify_gfx_mpsrt_op(this->getOperation(), ConcreteType::getOperationName());
    }
};

class Conv2DOp final : public GfxMpsrtOp<Conv2DOp> {
public:
    using GfxMpsrtOp<Conv2DOp>::GfxMpsrtOp;
    static llvm::StringRef getOperationName() {
        return "gfx.mpsrt.conv2d";
    }
};

class GroupConv2DOp final : public GfxMpsrtOp<GroupConv2DOp> {
public:
    using GfxMpsrtOp<GroupConv2DOp>::GfxMpsrtOp;
    static llvm::StringRef getOperationName() {
        return "gfx.mpsrt.group_conv2d";
    }
};

class Pool2DOp final : public GfxMpsrtOp<Pool2DOp> {
public:
    using GfxMpsrtOp<Pool2DOp>::GfxMpsrtOp;
    static llvm::StringRef getOperationName() {
        return "gfx.mpsrt.pool2d";
    }
};

class Resize2DOp final : public GfxMpsrtOp<Resize2DOp> {
public:
    using GfxMpsrtOp<Resize2DOp>::GfxMpsrtOp;
    static llvm::StringRef getOperationName() {
        return "gfx.mpsrt.resize2d";
    }
};

class GemmOp final : public GfxMpsrtOp<GemmOp> {
public:
    using GfxMpsrtOp<GemmOp>::GfxMpsrtOp;
    static llvm::StringRef getOperationName() {
        return "gfx.mpsrt.gemm";
    }
};

class SoftmaxOp final : public GfxMpsrtOp<SoftmaxOp> {
public:
    using GfxMpsrtOp<SoftmaxOp>::GfxMpsrtOp;
    static llvm::StringRef getOperationName() {
        return "gfx.mpsrt.softmax";
    }
};

class TopKOp final : public GfxMpsrtOp<TopKOp> {
public:
    using GfxMpsrtOp<TopKOp>::GfxMpsrtOp;
    static llvm::StringRef getOperationName() {
        return "gfx.mpsrt.topk";
    }
};

class SdpaOp final : public GfxMpsrtOp<SdpaOp> {
public:
    using GfxMpsrtOp<SdpaOp>::GfxMpsrtOp;
    static llvm::StringRef getOperationName() {
        return "gfx.mpsrt.sdpa";
    }
};

class ToImageOp final : public GfxMpsrtOp<ToImageOp> {
public:
    using GfxMpsrtOp<ToImageOp>::GfxMpsrtOp;
    static llvm::StringRef getOperationName() {
        return "gfx.mpsrt.to_image";
    }
};

class ToMatrixOp final : public GfxMpsrtOp<ToMatrixOp> {
public:
    using GfxMpsrtOp<ToMatrixOp>::GfxMpsrtOp;
    static llvm::StringRef getOperationName() {
        return "gfx.mpsrt.to_matrix";
    }
};

class ToNDArrayOp final : public GfxMpsrtOp<ToNDArrayOp> {
public:
    using GfxMpsrtOp<ToNDArrayOp>::GfxMpsrtOp;
    static llvm::StringRef getOperationName() {
        return "gfx.mpsrt.to_ndarray";
    }
};

class ToBufferOp final : public GfxMpsrtOp<ToBufferOp> {
public:
    using GfxMpsrtOp<ToBufferOp>::GfxMpsrtOp;
    static llvm::StringRef getOperationName() {
        return "gfx.mpsrt.to_buffer";
    }
};

class AliasOp final : public GfxMpsrtOp<AliasOp> {
public:
    using GfxMpsrtOp<AliasOp>::GfxMpsrtOp;
    static llvm::StringRef getOperationName() {
        return "gfx.mpsrt.alias";
    }
};

class DispatchOp final : public GfxMpsrtOp<DispatchOp> {
public:
    using GfxMpsrtOp<DispatchOp>::GfxMpsrtOp;
    static llvm::StringRef getOperationName() {
        return "gfx.mpsrt.dispatch";
    }
};

class ReturnOp final : public GfxMpsrtOp<ReturnOp> {
public:
    using GfxMpsrtOp<ReturnOp>::GfxMpsrtOp;
    static llvm::StringRef getOperationName() {
        return "gfx.mpsrt.return";
    }
};

void register_gfx_mpsrt_dialect(mlir::MLIRContext& context);

}  // namespace mpsrt
}  // namespace gfx_plugin
}  // namespace ov
