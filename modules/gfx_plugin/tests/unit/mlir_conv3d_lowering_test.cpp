// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "mlir/mlir_builder.hpp"
#include "mlir/mlir_passes.hpp"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

TEST(GfxMlirTransforms, Conv3DLoweringExplicitPadding) {
#if !ENABLE_GFX_MLIR
    GTEST_SKIP() << "GFX MLIR is disabled";
#else
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                         ov::Shape{1, 3, 5, 5, 5});
    std::vector<float> weights_data(4 * 3 * 3 * 3 * 3, 0.1f);
    auto weights = ov::op::v0::Constant::create(ov::element::f32,
                                                ov::Shape{4, 3, 3, 3, 3},
                                                weights_data);
    auto conv = std::make_shared<ov::op::v1::Convolution>(
        input,
        weights,
        ov::Strides{1, 1, 1},
        ov::CoordinateDiff{1, 1, 1},
        ov::CoordinateDiff{1, 1, 1},
        ov::Strides{1, 1, 1});
    auto res = std::make_shared<ov::op::v0::Result>(conv);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                             ov::ParameterVector{input},
                                             "conv3d_padding_test");

    mlir::MLIRContext ctx;
    auto module = ov::gfx_plugin::build_mlir_conv3d_from_model(model, ctx);
    ASSERT_TRUE(module) << "Failed to build MLIR module for Conv3D";

    ASSERT_NO_THROW(ov::gfx_plugin::run_mlir_pipeline(module,
                                                      /*use_alloca=*/false,
                                                      /*use_parallel_loops=*/false));

    bool has_conv = false;
    bool has_for = false;
    bool has_rank5_alloc = false;
    module.walk([&](mlir::Operation* op) {
        if (llvm::isa<mlir::linalg::Conv3DNcdhwFcdhwOp>(op)) {
            has_conv = true;
        }
        if (llvm::isa<mlir::scf::ForOp>(op)) {
            has_for = true;
        }
        if (auto alloc = mlir::dyn_cast<mlir::memref::AllocOp>(op)) {
            auto ty = alloc.getType();
            if (ty && ty.getRank() == 5) {
                has_rank5_alloc = true;
            }
        }
    });
    EXPECT_FALSE(has_conv) << "Conv3D should be lowered before SPIR-V";
    EXPECT_TRUE(has_for) << "Expected scf.for after lowering";
    EXPECT_FALSE(has_rank5_alloc) << "Padding alloc should be eliminated";
#endif
}
