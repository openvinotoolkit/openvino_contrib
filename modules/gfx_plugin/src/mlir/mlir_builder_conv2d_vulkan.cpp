// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "openvino/op/convolution.hpp"

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
            OPENVINO_THROW("Conv2D Vulkan builder: unsupported element type ", element_type);
    }
}

mlir::MemRefType to_memref_type(const ov::PartialShape& pshape,
                                ov::element::Type element_type,
                                mlir::MLIRContext& ctx) {
    OPENVINO_ASSERT(pshape.rank().is_static(), "Conv2D Vulkan builder: dynamic rank is not supported");
    mlir::SmallVector<int64_t> dims;
    dims.reserve(pshape.rank().get_length());
    for (const auto& dim : pshape) {
        OPENVINO_ASSERT(dim.is_static(), "Conv2D Vulkan builder: dynamic dim is not supported");
        dims.push_back(static_cast<int64_t>(dim.get_length()));
    }
    return mlir::MemRefType::get(dims, to_elem_type(element_type, ctx));
}

}  // namespace

mlir::ModuleOp build_mlir_conv2d_vulkan(const std::shared_ptr<const ov::op::v1::Convolution>& conv,
                                        mlir::MLIRContext& ctx) {
    OPENVINO_ASSERT(conv, "Conv2D Vulkan builder: null op");

    const auto& in_pshape = conv->get_input_partial_shape(0);
    const auto& w_pshape = conv->get_input_partial_shape(1);
    const auto& out_pshape = conv->get_output_partial_shape(0);
    OPENVINO_ASSERT(in_pshape.rank().is_static() && in_pshape.rank().get_length() == 4,
                    "Conv2D Vulkan builder expects rank-4 input");
    OPENVINO_ASSERT(w_pshape.rank().is_static() && w_pshape.rank().get_length() == 4,
                    "Conv2D Vulkan builder expects rank-4 weights");
    OPENVINO_ASSERT(out_pshape.rank().is_static() && out_pshape.rank().get_length() == 4,
                    "Conv2D Vulkan builder expects rank-4 output");

    ctx.loadDialect<mlir::func::FuncDialect,
                    mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect,
                    mlir::arith::ArithDialect,
                    mlir::math::MathDialect>();

    auto input_ty = to_memref_type(in_pshape, conv->get_input_element_type(0), ctx);
    auto weight_ty = to_memref_type(w_pshape, conv->get_input_element_type(1), ctx);
    auto output_ty = to_memref_type(out_pshape, conv->get_output_element_type(0), ctx);
    auto elem_ty = output_ty.getElementType();
    mlir::Type compute_ty = elem_ty;
    const auto& in_shape = conv->get_input_shape(0);
    const auto& w_shape = conv->get_input_shape(1);
    const auto& out_shape = conv->get_output_shape(0);
    const auto& pads_begin = conv->get_pads_begin();
    const auto& strides = conv->get_strides();
    const auto& dilations = conv->get_dilations();

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({input_ty, weight_ty, output_ty}, {});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "conv2d_main", func_type);
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
    auto zero = b.create<mlir::arith::ConstantOp>(loc, mlir::FloatAttr::get(compute_ty, 0.0));
    auto n_dim = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(out_shape.at(0)));
    auto c_in = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(in_shape.at(1)));
    auto in_h = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(in_shape.at(2)));
    auto in_w = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(in_shape.at(3)));
    auto c_out = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(w_shape.at(0)));
    auto k_h = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(w_shape.at(2)));
    auto k_w = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(w_shape.at(3)));
    auto stride_h = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(strides.at(0)));
    auto stride_w = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(strides.at(1)));
    auto dil_h = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(dilations.at(0)));
    auto dil_w = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(dilations.at(1)));
    auto pad_top = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(pads_begin.at(0)));
    auto pad_left = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(pads_begin.at(1)));
    auto out_h = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(out_shape.at(2)));
    auto out_w = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(out_shape.at(3)));
    auto input = func.getArgument(0);
    auto weight = func.getArgument(1);
    auto output = func.getArgument(2);

    auto for_n = b.create<mlir::scf::ForOp>(loc, c0, n_dim, c1);
    auto b_n = mlir::OpBuilder::atBlockBegin(for_n.getBody());
    auto for_oc = b_n.create<mlir::scf::ForOp>(loc, c0, c_out, c1);
    auto b_oc = mlir::OpBuilder::atBlockBegin(for_oc.getBody());
    auto for_oh = b_oc.create<mlir::scf::ForOp>(loc, c0, out_h, c1);
    auto b_oh = mlir::OpBuilder::atBlockBegin(for_oh.getBody());
    auto for_ow = b_oh.create<mlir::scf::ForOp>(loc, c0, out_w, c1);
    auto b_ow = mlir::OpBuilder::atBlockBegin(for_ow.getBody());
    auto for_ic = b_ow.create<mlir::scf::ForOp>(loc, c0, c_in, c1, mlir::ValueRange{zero.getResult()});
    auto b_ic = mlir::OpBuilder::atBlockBegin(for_ic.getBody());
    auto for_kh = b_ic.create<mlir::scf::ForOp>(loc,
                                                c0,
                                                k_h,
                                                c1,
                                                mlir::ValueRange{for_ic.getRegionIterArgs()[0]});
    auto b_kh = mlir::OpBuilder::atBlockBegin(for_kh.getBody());
    auto for_kw = b_kh.create<mlir::scf::ForOp>(loc,
                                                c0,
                                                k_w,
                                                c1,
                                                mlir::ValueRange{for_kh.getRegionIterArgs()[0]});
    auto b_kw = mlir::OpBuilder::atBlockBegin(for_kw.getBody());

    mlir::Value ih = b_kw.create<mlir::arith::AddIOp>(
        loc,
        b_kw.create<mlir::arith::MulIOp>(loc, for_oh.getInductionVar(), stride_h),
        b_kw.create<mlir::arith::MulIOp>(loc, for_kh.getInductionVar(), dil_h));
    ih = b_kw.create<mlir::arith::SubIOp>(loc, ih, pad_top);

    mlir::Value iw = b_kw.create<mlir::arith::AddIOp>(
        loc,
        b_kw.create<mlir::arith::MulIOp>(loc, for_ow.getInductionVar(), stride_w),
        b_kw.create<mlir::arith::MulIOp>(loc, for_kw.getInductionVar(), dil_w));
    iw = b_kw.create<mlir::arith::SubIOp>(loc, iw, pad_left);

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
            mlir::ValueRange{for_n.getInductionVar(), for_ic.getInductionVar(), ih, iw});
        auto weight_val = then_builder.create<mlir::memref::LoadOp>(
            loc,
            weight,
            mlir::ValueRange{for_oc.getInductionVar(),
                             for_ic.getInductionVar(),
                             for_kh.getInductionVar(),
                             for_kw.getInductionVar()});
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

    if (!for_ic.getBody()->empty() && for_ic.getBody()->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        for_ic.getBody()->back().erase();
    }
    auto b_ic_end = mlir::OpBuilder::atBlockEnd(for_ic.getBody());
    b_ic_end.create<mlir::scf::YieldOp>(loc, for_kh.getResult(0));

    b_ow.setInsertionPoint(for_ow.getBody()->getTerminator());
    b_ow.create<mlir::memref::StoreOp>(loc,
                                       cast_to_output(b_ow, for_ic.getResult(0)),
                                       output,
                                       mlir::ValueRange{for_n.getInductionVar(),
                                                        for_oc.getInductionVar(),
                                                        for_oh.getInductionVar(),
                                                        for_ow.getInductionVar()});

    b.create<mlir::func::ReturnOp>(loc);
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
