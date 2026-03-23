// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "mlir/mlir_builder.hpp"
#include "mlir/mlir_passes.hpp"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

TEST(GfxMlirTransforms, Conv2DParallelLowering) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                         ov::Shape{1, 3, 8, 8});
    std::vector<float> weights_data(8 * 3 * 3 * 3, 0.1f);
    auto weights = ov::op::v0::Constant::create(ov::element::f32,
                                                ov::Shape{8, 3, 3, 3},
                                                weights_data);
    auto conv = std::make_shared<ov::op::v1::Convolution>(
        input,
        weights,
        ov::Strides{1, 1},
        ov::CoordinateDiff{1, 1},
        ov::CoordinateDiff{1, 1},
        ov::Strides{1, 1});
    auto res = std::make_shared<ov::op::v0::Result>(conv);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                             ov::ParameterVector{input},
                                             "conv2d_parallel_test");

    mlir::MLIRContext ctx;
    auto module = ov::gfx_plugin::build_mlir_conv2d_from_model(model, ctx);
    ASSERT_TRUE(module) << "Failed to build MLIR module for Conv2D";

    ASSERT_NO_THROW(ov::gfx_plugin::run_mlir_pipeline(module,
                                                      /*use_alloca=*/false,
                                                      /*use_parallel_loops=*/true));

    bool has_conv = false;
    bool has_parallel = false;
    module.walk([&](mlir::Operation* op) {
        if (llvm::isa<mlir::linalg::Conv2DNchwFchwOp>(op)) {
            has_conv = true;
        }
        if (llvm::isa<mlir::scf::ParallelOp>(op)) {
            has_parallel = true;
        }
    });
    EXPECT_FALSE(has_conv) << "Conv2D should be lowered before SPIR-V";
    EXPECT_TRUE(has_parallel) << "Expected scf.parallel after lowering";
}
