// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "mlir/mlir_builder.hpp"
#include "mlir/mlir_passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "openvino/core/model.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace {

mlir::Value strip_memref_casts(mlir::Value value) {
    while (auto cast = value.getDefiningOp<mlir::memref::CastOp>()) {
        value = cast.getSource();
    }
    while (auto cast = value.getDefiningOp<mlir::memref::ReinterpretCastOp>()) {
        value = cast.getSource();
    }
    return value;
}

}  // namespace

TEST(GfxMlirTransforms, LinearMatMulLoweringAccumulatesBeforeSingleStore) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 4});
    auto mm = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, false);
    auto res = std::make_shared<ov::op::v0::Result>(mm);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                             ov::ParameterVector{lhs, rhs},
                                             "linear_matmul_lowering");

    mlir::MLIRContext ctx;
    auto module = ov::gfx_plugin::build_mlir_module_from_model(model, ctx);
    ASSERT_TRUE(module);

    module->setAttr("gfx.linear_matmul_parallel", mlir::BoolAttr::get(&ctx, true));
    module->setAttr("gfx.skip_matmul_parallel", mlir::BoolAttr::get(&ctx, false));
    module->setAttr("gfx.prefer_parallel", mlir::BoolAttr::get(&ctx, false));
    module->setAttr("gfx.dispatch_threads_h",
                    mlir::IntegerAttr::get(mlir::IndexType::get(&ctx), 1));
    module->setAttr("gfx.dispatch_threads_w",
                    mlir::IntegerAttr::get(mlir::IndexType::get(&ctx), 32));

    ASSERT_NO_THROW(ov::gfx_plugin::run_mlir_pipeline(module,
                                                      /*use_alloca=*/false,
                                                      /*use_parallel_loops=*/false));

    auto func = module.lookupSymbol<mlir::func::FuncOp>("matmul_main");
    ASSERT_TRUE(func);
    ASSERT_EQ(func.getNumArguments(), 3u);
    auto output_arg = func.getArgument(2);

    size_t output_store_count = 0;
    size_t output_load_count = 0;
    func.walk([&](mlir::memref::StoreOp op) {
        if (strip_memref_casts(op.getMemRef()) == output_arg) {
            ++output_store_count;
        }
    });
    func.walk([&](mlir::memref::LoadOp op) {
        if (strip_memref_casts(op.getMemRef()) == output_arg) {
            ++output_load_count;
        }
    });

    EXPECT_EQ(output_store_count, 1u);
    EXPECT_EQ(output_load_count, 0u);
}

TEST(GfxMlirTransforms, LinearMatMulLoweringKeepsLargeKSmallMBatchMatMulSerial) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 400, 400});
    auto mm = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, true);
    auto res = std::make_shared<ov::op::v0::Result>(mm);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                             ov::ParameterVector{lhs, rhs},
                                             "linear_batched_small_m_large_k_matmul");

    mlir::MLIRContext ctx;
    auto module = ov::gfx_plugin::build_mlir_module_from_model(model, ctx);
    ASSERT_TRUE(module);

    module->setAttr("gfx.linear_matmul_parallel", mlir::BoolAttr::get(&ctx, true));
    module->setAttr("gfx.skip_matmul_parallel", mlir::BoolAttr::get(&ctx, false));
    module->setAttr("gfx.prefer_parallel", mlir::BoolAttr::get(&ctx, false));
    module->setAttr("gfx.dispatch_threads_h",
                    mlir::IntegerAttr::get(mlir::IndexType::get(&ctx), 1));
    module->setAttr("gfx.dispatch_threads_w",
                    mlir::IntegerAttr::get(mlir::IndexType::get(&ctx), 32));

    ASSERT_NO_THROW(ov::gfx_plugin::run_mlir_pipeline(module,
                                                      /*use_alloca=*/false,
                                                      /*use_parallel_loops=*/false));

    auto func = module.lookupSymbol<mlir::func::FuncOp>("matmul_main");
    ASSERT_TRUE(func);

    size_t parallel_count = 0;
    size_t serial_for_count = 0;
    func.walk([&](mlir::scf::ParallelOp) {
        ++parallel_count;
    });
    func.walk([&](mlir::scf::ForOp) {
        ++serial_for_count;
    });

    EXPECT_EQ(parallel_count, 0u);
    EXPECT_GT(serial_for_count, 0u);
}
