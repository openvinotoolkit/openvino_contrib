// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "mlir/codegen_common.hpp"
#include "mlir/mlir_builder.hpp"
#include "mlir/mlir_passes.hpp"
#include "transforms/conv_parallel_lowering.hpp"
#include "transforms/conv_im2col_matmul_rewrite.hpp"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/Support/raw_ostream.h"

#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"

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

TEST(GfxMlirTransforms, Conv2DInteriorTileInputWindowCheckIsStrictlyBoundsSafe) {
    EXPECT_FALSE(ov::gfx_plugin::detail::is_conv_tile_input_h_interior(
        /*oh_base=*/0,
        /*tile_h=*/8,
        /*stride_h=*/1,
        /*dil_h=*/1,
        /*kernel_h=*/3,
        /*pad_h=*/1,
        /*input_h=*/80));
    EXPECT_TRUE(ov::gfx_plugin::detail::is_conv_tile_input_h_interior(
        /*oh_base=*/8,
        /*tile_h=*/8,
        /*stride_h=*/1,
        /*dil_h=*/1,
        /*kernel_h=*/3,
        /*pad_h=*/1,
        /*input_h=*/80));
    EXPECT_FALSE(ov::gfx_plugin::detail::is_conv_tile_input_w_interior(
        /*ow_base=*/0,
        /*tile_w=*/8,
        /*stride_w=*/1,
        /*dil_w=*/1,
        /*kernel_w=*/3,
        /*pad_w=*/1,
        /*input_w=*/80));
    EXPECT_TRUE(ov::gfx_plugin::detail::is_conv_tile_input_w_interior(
        /*ow_base=*/8,
        /*tile_w=*/8,
        /*stride_w=*/1,
        /*dil_w=*/1,
        /*kernel_w=*/3,
        /*pad_w=*/1,
        /*input_w=*/80));
    EXPECT_FALSE(ov::gfx_plugin::detail::is_conv_tile_input_interior(
        /*oh_base=*/0,
        /*ow_base=*/0,
        /*tile_h=*/8,
        /*tile_w=*/8,
        /*stride_h=*/1,
        /*stride_w=*/1,
        /*dil_h=*/1,
        /*dil_w=*/1,
        /*kernel_h=*/3,
        /*kernel_w=*/3,
        /*pad_h=*/1,
        /*pad_w=*/1,
        /*input_h=*/80,
        /*input_w=*/80));
    EXPECT_TRUE(ov::gfx_plugin::detail::is_conv_tile_input_interior(
        /*oh_base=*/8,
        /*ow_base=*/8,
        /*tile_h=*/8,
        /*tile_w=*/8,
        /*stride_h=*/1,
        /*stride_w=*/1,
        /*dil_h=*/1,
        /*dil_w=*/1,
        /*kernel_h=*/3,
        /*kernel_w=*/3,
        /*pad_h=*/1,
        /*pad_w=*/1,
        /*input_h=*/80,
        /*input_w=*/80));
    EXPECT_FALSE(ov::gfx_plugin::detail::is_conv_tile_input_interior(
        /*oh_base=*/0,
        /*ow_base=*/0,
        /*tile_h=*/8,
        /*tile_w=*/8,
        /*stride_h=*/2,
        /*stride_w=*/2,
        /*dil_h=*/1,
        /*dil_w=*/1,
        /*kernel_h=*/3,
        /*kernel_w=*/3,
        /*pad_h=*/1,
        /*pad_w=*/1,
        /*input_h=*/160,
        /*input_w=*/160));
    EXPECT_TRUE(ov::gfx_plugin::detail::is_conv_tile_input_interior(
        /*oh_base=*/8,
        /*ow_base=*/8,
        /*tile_h=*/8,
        /*tile_w=*/8,
        /*stride_h=*/2,
        /*stride_w=*/2,
        /*dil_h=*/1,
        /*dil_w=*/1,
        /*kernel_h=*/3,
        /*kernel_w=*/3,
        /*pad_h=*/1,
        /*pad_w=*/1,
        /*input_h=*/160,
        /*input_w=*/160));
    EXPECT_TRUE(ov::gfx_plugin::detail::is_conv_tile_input_interior(
        /*oh_base=*/15,
        /*ow_base=*/15,
        /*tile_h=*/4,
        /*tile_w=*/4,
        /*stride_h=*/1,
        /*stride_w=*/1,
        /*dil_h=*/1,
        /*dil_w=*/1,
        /*kernel_h=*/3,
        /*kernel_w=*/3,
        /*pad_h=*/1,
        /*pad_w=*/1,
        /*input_h=*/20,
        /*input_w=*/20));
}

TEST(GfxMlirTransforms, Conv2DBuilderUsesCanonicalConvOp) {
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
                                             "conv2d_builder_canonical");

    mlir::MLIRContext ctx;
    auto module = ov::gfx_plugin::build_mlir_conv2d_from_model(model, ctx);
    ASSERT_TRUE(module) << "Failed to build MLIR module for Conv2D";

    bool has_conv = false;
    bool has_pad = false;
    module.walk([&](mlir::Operation* op) {
        has_conv = has_conv || llvm::isa<mlir::linalg::Conv2DNchwFchwOp>(op);
        has_pad = has_pad || llvm::isa<mlir::tensor::PadOp>(op);
    });

    EXPECT_TRUE(has_conv) << "Conv2D builder should emit canonical linalg Conv op";
    EXPECT_TRUE(has_pad) << "Conv2D builder should materialize explicit padding in MLIR";
}

TEST(GfxMlirTransforms, Conv2DIm2ColRewriteLowersAuxiliaryStages) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                         ov::Shape{1, 64, 40, 40});
    std::vector<float> weights_data(64 * 64 * 3 * 3, 0.1f);
    auto weights = ov::op::v0::Constant::create(ov::element::f32,
                                                ov::Shape{64, 64, 3, 3},
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
                                             "conv2d_im2col_test");

    mlir::MLIRContext ctx;
    auto module = ov::gfx_plugin::build_mlir_conv2d_from_model(model, ctx);
    ASSERT_TRUE(module) << "Failed to build MLIR module for Conv2D";
    module->setAttr("gfx.conv_algorithm_kind", mlir::StringAttr::get(&ctx, "im2col_matmul"));

    ASSERT_NO_THROW(ov::gfx_plugin::run_mlir_pipeline(module,
                                                      /*use_alloca=*/false,
                                                      /*use_parallel_loops=*/true));

    bool has_tagged_generic = false;
    bool has_parallel = false;
    module.walk([&](mlir::Operation* op) {
        if (op->getAttr("gfx.im2col_stage") != nullptr) {
            has_tagged_generic = true;
        }
        if (llvm::isa<mlir::scf::ParallelOp>(op)) {
            has_parallel = true;
        }
    });

    EXPECT_FALSE(has_tagged_generic) << "Auxiliary im2col generics should be lowered before codegen";
    EXPECT_TRUE(has_parallel) << "Expected scf.parallel after im2col lowering";
}

TEST(GfxMlirTransforms, Conv2DIm2ColRewriteUsesPlainMatmulForBatchOne) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                         ov::Shape{1, 64, 40, 40});
    std::vector<float> weights_data(64 * 64 * 3 * 3, 0.1f);
    auto weights = ov::op::v0::Constant::create(ov::element::f32,
                                                ov::Shape{64, 64, 3, 3},
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
                                             "conv2d_im2col_plain_matmul");

    mlir::MLIRContext ctx;
    auto module = ov::gfx_plugin::build_mlir_conv2d_from_model(model, ctx);
    ASSERT_TRUE(module) << "Failed to build MLIR module for Conv2D";
    module->setAttr("gfx.conv_algorithm_kind", mlir::StringAttr::get(&ctx, "im2col_matmul"));

    ov::gfx_plugin::run_conv_im2col_matmul_rewrite(module);

    bool has_plain_matmul = false;
    bool has_batch_matmul = false;
    bool has_expand_shape = false;
    bool has_restore_output = false;
    module.walk([&](mlir::Operation* op) {
        has_plain_matmul = has_plain_matmul || llvm::isa<mlir::linalg::MatmulOp>(op);
        has_batch_matmul = has_batch_matmul || llvm::isa<mlir::linalg::BatchMatmulOp>(op);
        has_expand_shape = has_expand_shape || llvm::isa<mlir::tensor::ExpandShapeOp>(op);
        if (auto attr = op->getAttrOfType<mlir::StringAttr>("gfx.im2col_stage")) {
            has_restore_output = has_restore_output || attr.getValue() == "restore_output";
        }
    });

    EXPECT_TRUE(has_plain_matmul) << "Batch-1 im2col route should use linalg.matmul";
    EXPECT_FALSE(has_batch_matmul) << "Batch-1 im2col route should avoid batch matmul";
    EXPECT_TRUE(has_expand_shape) << "Default batch-1 im2col route should use view-like expand_shape";
    EXPECT_FALSE(has_restore_output) << "Default batch-1 im2col route should avoid restore_output stage";
}

TEST(GfxMlirTransforms, VulkanConv2DBuilderUsesParallelGpuLaunchForBatchOne) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                         ov::Shape{1, 128, 40, 40});
    std::vector<float> weights_data(256 * 128 * 3 * 3, 0.1f);
    auto weights = ov::op::v0::Constant::create(ov::element::f32,
                                                ov::Shape{256, 128, 3, 3},
                                                weights_data);
    auto conv = std::make_shared<ov::op::v1::Convolution>(
        input,
        weights,
        ov::Strides{2, 2},
        ov::CoordinateDiff{1, 1},
        ov::CoordinateDiff{1, 1},
        ov::Strides{1, 1});

    ov::gfx_plugin::ParallelDispatchConfig dispatch{};
    dispatch.enabled = true;
    dispatch.tile_h = 8;
    dispatch.tile_w = 8;
    dispatch.threads_h = 8;
    dispatch.threads_w = 8;

    mlir::MLIRContext ctx;
    auto module = ov::gfx_plugin::build_mlir_conv2d_vulkan(conv, ctx, &dispatch);
    ASSERT_TRUE(module);

    auto func = module.lookupSymbol<mlir::func::FuncOp>("conv2d_main");
    ASSERT_TRUE(func);

    bool has_launch = false;
    bool has_gpu_func = false;
    bool has_thread_id = false;
    module.walk([&](mlir::gpu::LaunchFuncOp) {
        has_launch = true;
    });
    module.walk([&](mlir::gpu::GPUFuncOp) {
        has_gpu_func = true;
    });
    module.walk([&](mlir::gpu::ThreadIdOp) {
        has_thread_id = true;
    });

    auto parallel_dispatch = module->getAttrOfType<mlir::BoolAttr>("gfx.parallel_dispatch");
    auto threads_h = module->getAttrOfType<mlir::IntegerAttr>("gfx.dispatch_threads_h");
    auto threads_w = module->getAttrOfType<mlir::IntegerAttr>("gfx.dispatch_threads_w");
    ASSERT_TRUE(parallel_dispatch);
    ASSERT_TRUE(threads_h);
    ASSERT_TRUE(threads_w);
    EXPECT_TRUE(parallel_dispatch.getValue());
    EXPECT_EQ(threads_h.getInt(), 8);
    EXPECT_EQ(threads_w.getInt(), 8);
    EXPECT_TRUE(has_launch);
    EXPECT_TRUE(has_gpu_func);
    EXPECT_TRUE(has_thread_id);
}

TEST(GfxMlirTransforms, VulkanConv2DBuilderFallsBackToSerialForBatchGreaterThanOne) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                         ov::Shape{2, 64, 20, 20});
    std::vector<float> weights_data(64 * 64 * 3 * 3, 0.1f);
    auto weights = ov::op::v0::Constant::create(ov::element::f32,
                                                ov::Shape{64, 64, 3, 3},
                                                weights_data);
    auto conv = std::make_shared<ov::op::v1::Convolution>(
        input,
        weights,
        ov::Strides{1, 1},
        ov::CoordinateDiff{1, 1},
        ov::CoordinateDiff{1, 1},
        ov::Strides{1, 1});

    ov::gfx_plugin::ParallelDispatchConfig dispatch{};
    dispatch.enabled = true;
    dispatch.tile_h = 8;
    dispatch.tile_w = 8;
    dispatch.threads_h = 8;
    dispatch.threads_w = 8;

    mlir::MLIRContext ctx;
    auto module = ov::gfx_plugin::build_mlir_conv2d_vulkan(conv, ctx, &dispatch);
    ASSERT_TRUE(module);

    bool has_launch = false;
    bool has_parallel_dispatch_attr = false;
    module.walk([&](mlir::gpu::LaunchFuncOp) {
        has_launch = true;
    });
    if (auto attr = module->getAttrOfType<mlir::BoolAttr>("gfx.parallel_dispatch")) {
        has_parallel_dispatch_attr = attr.getValue();
    }

    EXPECT_FALSE(has_launch);
    EXPECT_FALSE(has_parallel_dispatch_attr);
}

TEST(GfxMlirTransforms, GroupConv2DBuilderSetsPadAttrsForGroupsOne) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                         ov::Shape{1, 3, 8, 8});
    std::vector<float> weights_data(1 * 8 * 3 * 3 * 3, 0.1f);
    auto weights = ov::op::v0::Constant::create(ov::element::f32,
                                                ov::Shape{1, 8, 3, 3, 3},
                                                weights_data);
    auto gconv = std::make_shared<ov::op::v1::GroupConvolution>(
        input,
        weights,
        ov::Strides{1, 1},
        ov::CoordinateDiff{1, 1},
        ov::CoordinateDiff{1, 1},
        ov::Strides{1, 1});
    auto res = std::make_shared<ov::op::v0::Result>(gconv);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                             ov::ParameterVector{input},
                                             "group_conv2d_pad_attrs");

    mlir::MLIRContext ctx;
    auto module = ov::gfx_plugin::build_mlir_group_conv2d_from_model(model, ctx);
    ASSERT_TRUE(module) << "Failed to build MLIR module for GroupConv2D";

    bool found_conv = false;
    module.walk([&](mlir::linalg::Conv2DNchwFchwOp op) {
        found_conv = true;
        auto pad_begin = op->getAttrOfType<mlir::DenseIntElementsAttr>("gfx.pad_begin");
        auto pad_end = op->getAttrOfType<mlir::DenseIntElementsAttr>("gfx.pad_end");
        ASSERT_TRUE(pad_begin);
        ASSERT_TRUE(pad_end);
        EXPECT_EQ(pad_begin.getNumElements(), 2u);
        EXPECT_EQ(pad_end.getNumElements(), 2u);
    });

    EXPECT_TRUE(found_conv) << "Expected canonical linalg conv op for groups==1 GroupConv2D";
}

TEST(GfxMlirTransforms, AddBuilderAbsorbsInputTransposeIntoAffineMap) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 8, 10, 16});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 16, 8, 10});
    auto perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(lhs, perm);
    auto add = std::make_shared<ov::op::v1::Add>(transpose, rhs);

    mlir::MLIRContext ctx;
    std::vector<ov::gfx_plugin::MlirInputTransformDesc> transforms(2);
    transforms[0].source_shape = lhs->get_shape();
    transforms[0].transpose_permutation = {0, 3, 1, 2};
    auto module = ov::gfx_plugin::build_mlir_add_from_node(add, ctx, transforms);
    ASSERT_TRUE(module);

    auto func = ov::gfx_plugin::get_entry_func(module);
    ASSERT_TRUE(func);
    auto arg0_ty = llvm::dyn_cast<mlir::RankedTensorType>(func.getArgument(0).getType());
    auto arg1_ty = llvm::dyn_cast<mlir::RankedTensorType>(func.getArgument(1).getType());
    ASSERT_TRUE(arg0_ty);
    ASSERT_TRUE(arg1_ty);
    EXPECT_EQ(arg0_ty.getShape()[1], 8);
    EXPECT_EQ(arg0_ty.getShape()[2], 10);
    EXPECT_EQ(arg0_ty.getShape()[3], 16);
    EXPECT_EQ(arg1_ty.getShape()[1], 16);
    EXPECT_EQ(arg1_ty.getShape()[2], 8);
    EXPECT_EQ(arg1_ty.getShape()[3], 10);

    bool found_generic = false;
    module.walk([&](mlir::linalg::GenericOp generic) {
        found_generic = true;
        auto maps = generic.getIndexingMapsArray();
        ASSERT_EQ(maps.size(), 3u);
        auto lhs_map = maps[0];
        std::string lhs_map_text;
        llvm::raw_string_ostream os(lhs_map_text);
        lhs_map.print(os);
        os.flush();
        EXPECT_EQ(lhs_map_text, "(d0, d1, d2, d3) -> (0, d2, d3, d1)");
    });

    EXPECT_TRUE(found_generic) << "Expected linalg.generic for absorbed transpose Add";
    ASSERT_TRUE(module->hasAttr("gfx.absorbed_input0_perm"));
}

TEST(GfxMlirTransforms, SplitBuilderAbsorbsInputTransposeIntoGeneric) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4, 2, 96, 400});
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
    auto lengths = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {32, 32, 32});
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(input, axis, lengths);

    mlir::MLIRContext ctx;
    ov::gfx_plugin::MlirInputTransformDesc transform;
    transform.source_shape = ov::Shape{4, 400, 2, 96};
    transform.transpose_permutation = {0, 2, 3, 1};
    auto module = ov::gfx_plugin::build_mlir_split_from_node(split,
                                                             ctx,
                                                             transform.source_shape,
                                                             &transform);
    ASSERT_TRUE(module);

    auto func = ov::gfx_plugin::get_entry_func(module);
    ASSERT_TRUE(func);
    auto arg0_ty = llvm::dyn_cast<mlir::RankedTensorType>(func.getArgument(0).getType());
    ASSERT_TRUE(arg0_ty);
    EXPECT_EQ(arg0_ty.getShape()[0], 4);
    EXPECT_EQ(arg0_ty.getShape()[1], 400);
    EXPECT_EQ(arg0_ty.getShape()[2], 2);
    EXPECT_EQ(arg0_ty.getShape()[3], 96);

    bool found_generic = false;
    module.walk([&](mlir::linalg::GenericOp generic) {
        found_generic = true;
    });
    EXPECT_TRUE(found_generic) << "Expected linalg.generic for absorbed transpose Split";
    ASSERT_TRUE(module->hasAttr("gfx.absorbed_input0_perm"));
    ASSERT_NO_THROW(ov::gfx_plugin::run_mlir_pipeline(module,
                                                      /*use_alloca=*/false,
                                                      /*use_parallel_loops=*/true));
}

TEST(GfxMlirTransforms, GroupConvBuilderAbsorbsInputTransposeIntoDepthwiseGeneric) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 64, 40, 40});
    std::vector<float> weights_data(64 * 1 * 1 * 7 * 7, 0.1f);
    auto weights = ov::op::v0::Constant::create(ov::element::f32,
                                                ov::Shape{64, 1, 1, 7, 7},
                                                weights_data);
    auto gconv = std::make_shared<ov::op::v1::GroupConvolution>(
        input,
        weights,
        ov::Strides{1, 1},
        ov::CoordinateDiff{3, 3},
        ov::CoordinateDiff{3, 3},
        ov::Strides{1, 1});

    mlir::MLIRContext ctx;
    ov::gfx_plugin::MlirInputTransformDesc transform;
    transform.source_shape = ov::Shape{1, 40, 40, 64};
    transform.transpose_permutation = {0, 3, 1, 2};
    auto module = ov::gfx_plugin::build_mlir_group_conv2d_from_node(gconv, ctx, &transform);
    ASSERT_TRUE(module);

    auto func = ov::gfx_plugin::get_entry_func(module);
    ASSERT_TRUE(func);
    auto arg0_ty = llvm::dyn_cast<mlir::RankedTensorType>(func.getArgument(0).getType());
    ASSERT_TRUE(arg0_ty);
    EXPECT_EQ(arg0_ty.getShape()[0], 1);
    EXPECT_EQ(arg0_ty.getShape()[1], 40);
    EXPECT_EQ(arg0_ty.getShape()[2], 40);
    EXPECT_EQ(arg0_ty.getShape()[3], 64);

    bool found_generic = false;
    module.walk([&](mlir::linalg::GenericOp generic) {
        found_generic = true;
    });
    EXPECT_TRUE(found_generic) << "Expected depthwise linalg.generic for absorbed transpose GroupConv";
    ASSERT_NO_THROW(ov::gfx_plugin::run_mlir_pipeline(module,
                                                      /*use_alloca=*/false,
                                                      /*use_parallel_loops=*/true));
}

TEST(GfxMlirTransforms, ConvBuilderAbsorbsInputTransposeIntoPointwiseGeneric) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 16, 4, 8400});
    std::vector<float> weights_data(1 * 16 * 1 * 1, 0.1f);
    auto weights = ov::op::v0::Constant::create(ov::element::f32,
                                                ov::Shape{1, 16, 1, 1},
                                                weights_data);
    auto conv = std::make_shared<ov::op::v1::Convolution>(
        input,
        weights,
        ov::Strides{1, 1},
        ov::CoordinateDiff{0, 0},
        ov::CoordinateDiff{0, 0},
        ov::Strides{1, 1});

    mlir::MLIRContext ctx;
    ov::gfx_plugin::MlirInputTransformDesc transform;
    transform.source_shape = ov::Shape{1, 4, 16, 8400};
    transform.transpose_permutation = {0, 2, 1, 3};
    auto module = ov::gfx_plugin::build_mlir_conv2d_from_node(conv, ctx, &transform);
    ASSERT_TRUE(module);

    auto func = ov::gfx_plugin::get_entry_func(module);
    ASSERT_TRUE(func);
    auto arg0_ty = llvm::dyn_cast<mlir::RankedTensorType>(func.getArgument(0).getType());
    ASSERT_TRUE(arg0_ty);
    EXPECT_EQ(arg0_ty.getShape()[0], 1);
    EXPECT_EQ(arg0_ty.getShape()[1], 4);
    EXPECT_EQ(arg0_ty.getShape()[2], 16);
    EXPECT_EQ(arg0_ty.getShape()[3], 8400);

    bool found_generic = false;
    module.walk([&](mlir::linalg::GenericOp generic) {
        found_generic = true;
    });
    EXPECT_TRUE(found_generic) << "Expected linalg.generic for absorbed transpose Conv";
    ASSERT_NO_THROW(ov::gfx_plugin::run_mlir_pipeline(module,
                                                      /*use_alloca=*/false,
                                                      /*use_parallel_loops=*/true));
}
