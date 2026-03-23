// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "openvino/op/group_conv.hpp"

namespace ov {
namespace gfx_plugin {

namespace {

mlir::Type to_elem_type(ov::element::Type element_type, mlir::MLIRContext& ctx) {
    switch (element_type) {
        case ov::element::f16:
            return mlir::Float16Type::get(&ctx);
        case ov::element::f32:
            return mlir::Float32Type::get(&ctx);
        default:
            OPENVINO_THROW("GroupConv2D Vulkan builder: unsupported element type ", element_type);
    }
}

mlir::MemRefType to_memref_type(const ov::PartialShape& pshape,
                                ov::element::Type element_type,
                                mlir::MLIRContext& ctx) {
    OPENVINO_ASSERT(pshape.rank().is_static(), "GroupConv2D Vulkan builder: dynamic rank is not supported");
    mlir::SmallVector<int64_t> dims;
    dims.reserve(pshape.rank().get_length());
    for (const auto& dim : pshape) {
        OPENVINO_ASSERT(dim.is_static(), "GroupConv2D Vulkan builder: dynamic dim is not supported");
        dims.push_back(static_cast<int64_t>(dim.get_length()));
    }
    return mlir::MemRefType::get(dims, to_elem_type(element_type, ctx));
}

}  // namespace

mlir::ModuleOp build_mlir_group_conv2d_vulkan(const std::shared_ptr<const ov::op::v1::GroupConvolution>& gconv,
                                              mlir::MLIRContext& ctx) {
    OPENVINO_ASSERT(gconv, "GroupConv2D Vulkan builder: null op");

    const auto& in_pshape = gconv->get_input_partial_shape(0);
    const auto& w_pshape = gconv->get_input_partial_shape(1);
    const auto& out_pshape = gconv->get_output_partial_shape(0);
    OPENVINO_ASSERT(in_pshape.rank().is_static() && in_pshape.rank().get_length() == 4,
                    "GroupConv2D Vulkan builder expects rank-4 input");
    OPENVINO_ASSERT(w_pshape.rank().is_static() && w_pshape.rank().get_length() == 5,
                    "GroupConv2D Vulkan builder expects rank-5 weights");
    OPENVINO_ASSERT(out_pshape.rank().is_static() && out_pshape.rank().get_length() == 4,
                    "GroupConv2D Vulkan builder expects rank-4 output");

    const auto& pads_begin = gconv->get_pads_begin();
    const auto& strides = gconv->get_strides();
    const auto& dilations = gconv->get_dilations();
    OPENVINO_ASSERT(pads_begin.size() == 2, "GroupConv2D Vulkan builder expects 2D padding");
    OPENVINO_ASSERT(strides.size() == 2, "GroupConv2D Vulkan builder expects 2D strides");
    OPENVINO_ASSERT(dilations.size() == 2, "GroupConv2D Vulkan builder expects 2D dilations");

    const int64_t groups = w_pshape[0].get_length();
    const int64_t out_c_pg = w_pshape[1].get_length();
    const int64_t in_c_pg = w_pshape[2].get_length();
    const int64_t kernel_h = w_pshape[3].get_length();
    const int64_t kernel_w = w_pshape[4].get_length();
    const int64_t in_c = in_pshape[1].get_length();
    const int64_t out_c = out_pshape[1].get_length();
    OPENVINO_ASSERT(groups == in_c && groups == out_c && out_c_pg == 1 && in_c_pg == 1,
                    "GroupConv2D Vulkan builder currently supports depthwise group convolution only");

    ctx.loadDialect<mlir::func::FuncDialect,
                    mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect,
                    mlir::arith::ArithDialect>();

    auto input_ty = to_memref_type(in_pshape, gconv->get_input_element_type(0), ctx);
    auto weight_ty = to_memref_type(w_pshape, gconv->get_input_element_type(1), ctx);
    auto output_ty = to_memref_type(out_pshape, gconv->get_output_element_type(0), ctx);
    auto elem_ty = output_ty.getElementType();
    mlir::Type compute_ty = elem_ty;

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({input_ty, weight_ty, output_ty}, {});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "group_conv2d_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    b.setInsertionPointToStart(&func.getBody().front());
    auto loc = mlir::UnknownLoc::get(&ctx);

    auto cast_to_compute = [&](mlir::OpBuilder& builder, mlir::Value value) -> mlir::Value {
        if (value.getType() == compute_ty) {
            return value;
        }
        return builder.create<mlir::arith::ExtFOp>(loc, compute_ty, value);
    };
    auto cast_to_output = [&](mlir::OpBuilder& builder, mlir::Value value) -> mlir::Value {
        if (value.getType() == elem_ty) {
            return value;
        }
        return builder.create<mlir::arith::TruncFOp>(loc, elem_ty, value);
    };

    auto c0 = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1 = b.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto pad_h = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(pads_begin[0]));
    auto pad_w = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(pads_begin[1]));
    auto stride_h = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(strides[0]));
    auto stride_w = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(strides[1]));
    auto dil_h = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(dilations[0]));
    auto dil_w = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(dilations[1]));

    auto input = func.getArgument(0);
    auto weight = func.getArgument(1);
    auto output = func.getArgument(2);

    auto out_n = b.create<mlir::memref::DimOp>(loc, output, 0);
    auto out_c_dim = b.create<mlir::memref::DimOp>(loc, output, 1);
    auto out_h = b.create<mlir::memref::DimOp>(loc, output, 2);
    auto out_w = b.create<mlir::memref::DimOp>(loc, output, 3);
    auto in_h = b.create<mlir::memref::DimOp>(loc, input, 2);
    auto in_w = b.create<mlir::memref::DimOp>(loc, input, 3);
    auto ker_h = b.create<mlir::arith::ConstantIndexOp>(loc, kernel_h);
    auto ker_w = b.create<mlir::arith::ConstantIndexOp>(loc, kernel_w);
    auto zero = b.create<mlir::arith::ConstantOp>(loc, mlir::FloatAttr::get(compute_ty, 0.0));

    auto for_n = b.create<mlir::scf::ForOp>(loc, c0, out_n, c1);
    auto b_n = mlir::OpBuilder::atBlockBegin(for_n.getBody());
    auto for_c = b_n.create<mlir::scf::ForOp>(loc, c0, out_c_dim, c1);
    auto b_c = mlir::OpBuilder::atBlockBegin(for_c.getBody());
    auto for_oh = b_c.create<mlir::scf::ForOp>(loc, c0, out_h, c1);
    auto b_oh = mlir::OpBuilder::atBlockBegin(for_oh.getBody());
    auto for_ow = b_oh.create<mlir::scf::ForOp>(loc, c0, out_w, c1);
    auto b_ow = mlir::OpBuilder::atBlockBegin(for_ow.getBody());

    auto for_kh = b_ow.create<mlir::scf::ForOp>(loc, c0, ker_h, c1, mlir::ValueRange{zero.getResult()});
    auto b_kh = mlir::OpBuilder::atBlockBegin(for_kh.getBody());
    auto for_kw =
        b_kh.create<mlir::scf::ForOp>(loc, c0, ker_w, c1, mlir::ValueRange{for_kh.getRegionIterArgs()[0]});
    auto b_kw = mlir::OpBuilder::atBlockBegin(for_kw.getBody());

    mlir::Value ih = b_kw.create<mlir::arith::AddIOp>(
        loc,
        b_kw.create<mlir::arith::MulIOp>(loc, for_oh.getInductionVar(), stride_h),
        b_kw.create<mlir::arith::MulIOp>(loc, for_kh.getInductionVar(), dil_h)).getResult();
    ih = b_kw.create<mlir::arith::SubIOp>(loc, ih, pad_h).getResult();

    mlir::Value iw = b_kw.create<mlir::arith::AddIOp>(
        loc,
        b_kw.create<mlir::arith::MulIOp>(loc, for_ow.getInductionVar(), stride_w),
        b_kw.create<mlir::arith::MulIOp>(loc, for_kw.getInductionVar(), dil_w)).getResult();
    iw = b_kw.create<mlir::arith::SubIOp>(loc, iw, pad_w).getResult();

    auto ih_ge0 = b_kw.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, ih, c0);
    auto ih_lt = b_kw.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, ih, in_h);
    auto iw_ge0 = b_kw.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, iw, c0);
    auto iw_lt = b_kw.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, iw, in_w);
    auto in_bounds = b_kw.create<mlir::arith::AndIOp>(loc, ih_ge0, ih_lt);
    in_bounds = b_kw.create<mlir::arith::AndIOp>(loc, in_bounds, iw_ge0);
    in_bounds = b_kw.create<mlir::arith::AndIOp>(loc, in_bounds, iw_lt);

    auto if_in_bounds =
        b_kw.create<mlir::scf::IfOp>(loc, mlir::TypeRange{compute_ty}, in_bounds, /*withElseRegion=*/true);
    {
        auto then_builder = if_in_bounds.getThenBodyBuilder();
        auto input_val = then_builder.create<mlir::memref::LoadOp>(
            loc,
            input,
            mlir::ValueRange{for_n.getInductionVar(), for_c.getInductionVar(), ih, iw});
        auto weight_val = then_builder.create<mlir::memref::LoadOp>(
            loc,
            weight,
            mlir::ValueRange{for_c.getInductionVar(), c0, c0, for_kh.getInductionVar(), for_kw.getInductionVar()});
        auto prod = then_builder.create<mlir::arith::MulFOp>(
            loc,
            cast_to_compute(then_builder, input_val),
            cast_to_compute(then_builder, weight_val));
        auto sum = then_builder.create<mlir::arith::AddFOp>(loc, prod, for_kw.getRegionIterArgs()[0]);
        then_builder.create<mlir::scf::YieldOp>(loc, sum.getResult());
    }
    {
        auto else_builder = if_in_bounds.getElseBodyBuilder();
        else_builder.create<mlir::scf::YieldOp>(loc, for_kw.getRegionIterArgs()[0]);
    }

    if (!for_kw.getBody()->empty() && for_kw.getBody()->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        for_kw.getBody()->back().erase();
    }
    auto b_kw_end = mlir::OpBuilder::atBlockEnd(for_kw.getBody());
    b_kw_end.create<mlir::scf::YieldOp>(loc, if_in_bounds.getResult(0));

    if (!for_kh.getBody()->empty() && for_kh.getBody()->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        for_kh.getBody()->back().erase();
    }
    auto b_kh_end = mlir::OpBuilder::atBlockEnd(for_kh.getBody());
    b_kh_end.create<mlir::scf::YieldOp>(loc, for_kw.getResult(0));

    auto out_value = cast_to_output(b_ow, for_kh.getResult(0));
    b_ow.setInsertionPoint(for_ow.getBody()->getTerminator());
    b_ow.create<mlir::memref::StoreOp>(
        loc,
        out_value,
        output,
        mlir::ValueRange{for_n.getInductionVar(),
                         for_c.getInductionVar(),
                         for_oh.getInductionVar(),
                         for_ow.getInductionVar()});

    b.create<mlir::func::ReturnOp>(loc);
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
