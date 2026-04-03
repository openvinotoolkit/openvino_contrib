// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_support.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Support/StorageUniquer.h"
#include "mlir/gfx_mlir_kernel_builder.hpp"

namespace ov {
namespace gfx_plugin {

bool mlir_supports_node(const std::shared_ptr<const ov::Node>& node) {
    if (!node) {
        return false;
    }
    auto& ctx = gfx_mlir_context();
    auto module = build_mlir_for_node(node, ctx);
    return module != nullptr;
}

mlir::MLIRContext& gfx_mlir_context() {
    static mlir::MLIRContext* ctx = []() {
        mlir::DialectRegistry registry;
        // Core IR + math/scf/memref/tensor/linalg/vector/affine/func/gpu/spirv.
        registry.insert<mlir::BuiltinDialect,
                        mlir::arith::ArithDialect,
                        mlir::math::MathDialect,
                        mlir::scf::SCFDialect,
                        mlir::affine::AffineDialect,
                        mlir::memref::MemRefDialect,
                        mlir::tensor::TensorDialect,
                        mlir::linalg::LinalgDialect,
                        mlir::vector::VectorDialect,
                        mlir::func::FuncDialect,
                        mlir::gpu::GPUDialect,
                        mlir::spirv::SPIRVDialect>();

        auto* c = new mlir::MLIRContext(registry);
        c->disableMultithreading();
        c->allowUnregisteredDialects();
        // Ensure all built-in and dependent dialects are fully registered (including attr storages).
        c->loadAllAvailableDialects();

        // Force-load dialects to register all parametric storages (e.g. DenseArrayAttr).
        c->getOrLoadDialect<mlir::BuiltinDialect>();
        c->getOrLoadDialect<mlir::arith::ArithDialect>();
        c->getOrLoadDialect<mlir::math::MathDialect>();
        c->getOrLoadDialect<mlir::scf::SCFDialect>();
        c->getOrLoadDialect<mlir::affine::AffineDialect>();
        c->getOrLoadDialect<mlir::memref::MemRefDialect>();
        c->getOrLoadDialect<mlir::tensor::TensorDialect>();
        c->getOrLoadDialect<mlir::linalg::LinalgDialect>();
        c->getOrLoadDialect<mlir::vector::VectorDialect>();
        c->getOrLoadDialect<mlir::func::FuncDialect>();
        c->getOrLoadDialect<mlir::gpu::GPUDialect>();
        c->getOrLoadDialect<mlir::spirv::SPIRVDialect>();

        return c;
    }();
    return *ctx;
}

}  // namespace gfx_plugin
}  // namespace ov
