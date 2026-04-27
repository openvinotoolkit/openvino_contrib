// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
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

void set_buffer_operand_attrs(mlir::ModuleOp module, mlir::OpBuilder& builder) {
    auto make_i32_array_attr = [&](std::initializer_list<int32_t> values) {
        mlir::SmallVector<mlir::Attribute, 8> attrs;
        attrs.reserve(values.size());
        for (int32_t value : values) {
            attrs.push_back(builder.getI32IntegerAttr(value));
        }
        return builder.getArrayAttr(attrs);
    };
    module->setAttr("gfx.fixed_arg_count", builder.getI32IntegerAttr(3));
    module->setAttr("gfx.kernel_operand_kinds", make_i32_array_attr({1, 1, 1}));
    module->setAttr("gfx.kernel_operand_arg_indices", make_i32_array_attr({0, 1, 2}));
}

void set_dispatch_attrs(mlir::ModuleOp module,
                        mlir::OpBuilder& builder,
                        const ParallelDispatchConfig& dispatch_cfg) {
    module->setAttr("gfx.parallel_dispatch", builder.getBoolAttr(dispatch_cfg.enabled));
    module->setAttr("gfx.dispatch_tile_h",
                    mlir::IntegerAttr::get(mlir::IndexType::get(module.getContext()),
                                           static_cast<int64_t>(dispatch_cfg.tile_h)));
    module->setAttr("gfx.dispatch_tile_w",
                    mlir::IntegerAttr::get(mlir::IndexType::get(module.getContext()),
                                           static_cast<int64_t>(dispatch_cfg.tile_w)));
    module->setAttr("gfx.dispatch_threads_h",
                    mlir::IntegerAttr::get(mlir::IndexType::get(module.getContext()),
                                           static_cast<int64_t>(dispatch_cfg.threads_h)));
    module->setAttr("gfx.dispatch_threads_w",
                    mlir::IntegerAttr::get(mlir::IndexType::get(module.getContext()),
                                           static_cast<int64_t>(dispatch_cfg.threads_w)));
}

mlir::ModuleOp build_serial_conv2d_vulkan(const std::shared_ptr<const ov::op::v1::Convolution>& conv,
                                          mlir::MLIRContext& ctx,
                                          mlir::MemRefType input_ty,
                                          mlir::MemRefType weight_ty,
                                          mlir::MemRefType output_ty) {
    const auto& in_shape = conv->get_input_shape(0);
    const auto& w_shape = conv->get_input_shape(1);
    const auto& out_shape = conv->get_output_shape(0);
    const auto& pads_begin = conv->get_pads_begin();
    const auto& strides = conv->get_strides();
    const auto& dilations = conv->get_dilations();

    auto elem_ty = output_ty.getElementType();
    mlir::Type compute_ty = elem_ty;
    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    set_buffer_operand_attrs(module, mb);

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

mlir::ModuleOp build_parallel_conv2d_vulkan(const std::shared_ptr<const ov::op::v1::Convolution>& conv,
                                            mlir::MLIRContext& ctx,
                                            mlir::MemRefType input_ty,
                                            mlir::MemRefType weight_ty,
                                            mlir::MemRefType output_ty,
                                            const ParallelDispatchConfig& dispatch_cfg) {
    const auto& in_shape = conv->get_input_shape(0);
    const auto& w_shape = conv->get_input_shape(1);
    const auto& out_shape = conv->get_output_shape(0);
    const auto& pads_begin = conv->get_pads_begin();

    auto elem_ty = output_ty.getElementType();
    mlir::Type compute_ty = elem_ty;
    mlir::OpBuilder b(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    b.setInsertionPointToStart(module.getBody());
    module->setAttr(mlir::gpu::GPUDialect::getContainerModuleAttrName(), b.getUnitAttr());
    set_buffer_operand_attrs(module, b);
    set_dispatch_attrs(module, b, dispatch_cfg);

    auto func_type = b.getFunctionType({input_ty, weight_ty, output_ty}, {});
    auto func = b.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "conv2d_main", func_type);
    func.addEntryBlock();

    auto gpu_mod = b.create<mlir::gpu::GPUModuleOp>(mlir::UnknownLoc::get(&ctx), "gfx_kernels");
    mlir::OpBuilder gpu_builder = mlir::OpBuilder::atBlockBegin(gpu_mod.getBody());
    auto gpu_func = gpu_builder.create<mlir::gpu::GPUFuncOp>(mlir::UnknownLoc::get(&ctx),
                                                             "conv2d_kernel",
                                                             func_type,
                                                             mlir::TypeRange{},
                                                             mlir::TypeRange{});
    gpu_func->setAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
    gpu_func.setKnownBlockSizeAttr(mlir::DenseI32ArrayAttr::get(
        &ctx,
        {static_cast<int32_t>(std::max<uint32_t>(1u, dispatch_cfg.threads_w)),
         static_cast<int32_t>(std::max<uint32_t>(1u, dispatch_cfg.threads_h)),
         1}));

    auto* entry = &gpu_func.getBody().front();
    mlir::OpBuilder body(entry, entry->begin());
    auto loc = gpu_func.getLoc();

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

    auto c0 = body.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1 = body.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto zero = body.create<mlir::arith::ConstantOp>(loc, mlir::FloatAttr::get(compute_ty, 0.0));
    auto one = body.create<mlir::arith::ConstantOp>(loc, mlir::FloatAttr::get(compute_ty, 1.0));
    auto c_in = body.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(in_shape.at(1)));
    auto in_h = body.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(in_shape.at(2)));
    auto in_w = body.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(in_shape.at(3)));
    auto c_out = body.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(w_shape.at(0)));
    auto k_h = body.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(w_shape.at(2)));
    auto k_w = body.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(w_shape.at(3)));
    auto stride_h = body.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(conv->get_strides().at(0)));
    auto stride_w = body.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(conv->get_strides().at(1)));
    auto dil_h = body.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(conv->get_dilations().at(0)));
    auto dil_w = body.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(conv->get_dilations().at(1)));
    auto pad_top = body.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(pads_begin.at(0)));
    auto pad_left = body.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(pads_begin.at(1)));
    auto out_h = body.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(out_shape.at(2)));
    auto out_w = body.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(out_shape.at(3)));
    auto in_h_last = body.create<mlir::arith::SubIOp>(loc, in_h, c1);
    auto in_w_last = body.create<mlir::arith::SubIOp>(loc, in_w, c1);
    auto bid_x = body.create<mlir::gpu::BlockIdOp>(loc, mlir::gpu::Dimension::x);
    auto bid_y = body.create<mlir::gpu::BlockIdOp>(loc, mlir::gpu::Dimension::y);
    auto bid_z = body.create<mlir::gpu::BlockIdOp>(loc, mlir::gpu::Dimension::z);
    auto bdim_x = body.create<mlir::gpu::BlockDimOp>(loc, mlir::gpu::Dimension::x);
    auto bdim_y = body.create<mlir::gpu::BlockDimOp>(loc, mlir::gpu::Dimension::y);
    auto tid_x = body.create<mlir::gpu::ThreadIdOp>(loc, mlir::gpu::Dimension::x);
    auto tid_y = body.create<mlir::gpu::ThreadIdOp>(loc, mlir::gpu::Dimension::y);
    auto n = c0;
    auto oc = bid_x;
    auto oh = body.create<mlir::arith::AddIOp>(loc, body.create<mlir::arith::MulIOp>(loc, bid_y, bdim_y), tid_y);
    auto ow = body.create<mlir::arith::AddIOp>(loc, body.create<mlir::arith::MulIOp>(loc, bid_z, bdim_x), tid_x);
    auto oc_valid = body.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, oc, c_out);
    auto oh_valid = body.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, oh, out_h);
    auto ow_valid = body.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, ow, out_w);
    auto active = body.create<mlir::arith::AndIOp>(loc, oc_valid, oh_valid);
    active = body.create<mlir::arith::AndIOp>(loc, active, ow_valid);
    auto active_if = body.create<mlir::scf::IfOp>(loc, active, /*withElseRegion=*/false);
    {
        auto then_builder = active_if.getThenBodyBuilder();
        auto acc = zero.getResult();
        auto base_ih = then_builder.create<mlir::arith::SubIOp>(
            loc, then_builder.create<mlir::arith::MulIOp>(loc, oh, stride_h), pad_top);
        auto base_iw = then_builder.create<mlir::arith::SubIOp>(
            loc, then_builder.create<mlir::arith::MulIOp>(loc, ow, stride_w), pad_left);
        auto for_ic = then_builder.create<mlir::scf::ForOp>(loc, c0, c_in, c1, mlir::ValueRange{acc});
        auto ic_builder = mlir::OpBuilder::atBlockBegin(for_ic.getBody());
        auto for_kh = ic_builder.create<mlir::scf::ForOp>(loc,
                                                          c0,
                                                          k_h,
                                                          c1,
                                                          mlir::ValueRange{for_ic.getRegionIterArgs()[0]});
        auto kh_builder = mlir::OpBuilder::atBlockBegin(for_kh.getBody());
        auto for_kw = kh_builder.create<mlir::scf::ForOp>(loc,
                                                          c0,
                                                          k_w,
                                                          c1,
                                                          mlir::ValueRange{for_kh.getRegionIterArgs()[0]});
        auto kw_builder = mlir::OpBuilder::atBlockBegin(for_kw.getBody());
        auto ih = kw_builder.create<mlir::arith::AddIOp>(
            loc,
            base_ih,
            kw_builder.create<mlir::arith::MulIOp>(loc, for_kh.getInductionVar(), dil_h));
        auto iw = kw_builder.create<mlir::arith::AddIOp>(
            loc,
            base_iw,
            kw_builder.create<mlir::arith::MulIOp>(loc, for_kw.getInductionVar(), dil_w));
        auto ih_ge0 = kw_builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, ih, c0);
        auto ih_lt = kw_builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, ih, in_h);
        auto iw_ge0 = kw_builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, iw, c0);
        auto iw_lt = kw_builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, iw, in_w);
        auto in_bounds = kw_builder.create<mlir::arith::AndIOp>(loc, ih_ge0, ih_lt);
        in_bounds = kw_builder.create<mlir::arith::AndIOp>(loc, in_bounds, iw_ge0);
        in_bounds = kw_builder.create<mlir::arith::AndIOp>(loc, in_bounds, iw_lt);
        auto ih_nonneg = kw_builder.create<mlir::arith::SelectOp>(loc, ih_ge0, ih, c0);
        auto iw_nonneg = kw_builder.create<mlir::arith::SelectOp>(loc, iw_ge0, iw, c0);
        auto ih_bounded =
            kw_builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, ih_nonneg, in_h);
        auto iw_bounded =
            kw_builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, iw_nonneg, in_w);
        auto ih_safe = kw_builder.create<mlir::arith::SelectOp>(loc, ih_bounded, ih_nonneg, in_h_last);
        auto iw_safe = kw_builder.create<mlir::arith::SelectOp>(loc, iw_bounded, iw_nonneg, in_w_last);
        auto input_val = kw_builder.create<mlir::memref::LoadOp>(
            loc,
            gpu_func.getArgument(0),
            mlir::ValueRange{n, for_ic.getInductionVar(), ih_safe, iw_safe});
        auto weight_val = kw_builder.create<mlir::memref::LoadOp>(
            loc,
            gpu_func.getArgument(1),
            mlir::ValueRange{oc, for_ic.getInductionVar(), for_kh.getInductionVar(), for_kw.getInductionVar()});
        auto input_comp = cast_to_compute(kw_builder, input_val);
        auto mask = kw_builder.create<mlir::arith::SelectOp>(loc, in_bounds, one, zero);
        auto masked_input = kw_builder.create<mlir::arith::MulFOp>(loc, input_comp, mask);
        auto prod = kw_builder.create<mlir::arith::MulFOp>(
            loc,
            masked_input,
            cast_to_compute(kw_builder, weight_val));
        auto sum = kw_builder.create<mlir::arith::AddFOp>(loc, prod, for_kw.getRegionIterArgs()[0]);
        if (!for_kw.getBody()->empty() && for_kw.getBody()->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
            for_kw.getBody()->back().erase();
        }
        auto kw_end = mlir::OpBuilder::atBlockEnd(for_kw.getBody());
        kw_end.create<mlir::scf::YieldOp>(loc, sum.getResult());
        if (!for_kh.getBody()->empty() && for_kh.getBody()->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
            for_kh.getBody()->back().erase();
        }
        auto kh_end = mlir::OpBuilder::atBlockEnd(for_kh.getBody());
        kh_end.create<mlir::scf::YieldOp>(loc, for_kw.getResult(0));
        if (!for_ic.getBody()->empty() && for_ic.getBody()->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
            for_ic.getBody()->back().erase();
        }
        auto ic_end = mlir::OpBuilder::atBlockEnd(for_ic.getBody());
        ic_end.create<mlir::scf::YieldOp>(loc, for_kh.getResult(0));
        then_builder.setInsertionPointAfter(for_ic);
        then_builder.create<mlir::memref::StoreOp>(loc,
                                                   cast_to_output(then_builder, for_ic.getResult(0)),
                                                   gpu_func.getArgument(2),
                                                   mlir::ValueRange{n, oc, oh, ow});
    }
    body.setInsertionPointAfter(active_if);
    body.create<mlir::gpu::ReturnOp>(loc);

    mlir::OpBuilder host_builder(func.getBody());
    host_builder.setInsertionPointToStart(&func.getBody().front());
    auto grid_x = host_builder.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(out_shape.at(1)));
    auto grid_y = host_builder.create<mlir::arith::ConstantIndexOp>(
        loc,
        static_cast<int64_t>((out_shape.at(2) + std::max<uint32_t>(1u, dispatch_cfg.threads_h) - 1) /
                             std::max<uint32_t>(1u, dispatch_cfg.threads_h)));
    auto grid_z = host_builder.create<mlir::arith::ConstantIndexOp>(
        loc,
        static_cast<int64_t>((out_shape.at(3) + std::max<uint32_t>(1u, dispatch_cfg.threads_w) - 1) /
                             std::max<uint32_t>(1u, dispatch_cfg.threads_w)));
    auto block_x = host_builder.create<mlir::arith::ConstantIndexOp>(
        loc, static_cast<int64_t>(std::max<uint32_t>(1u, dispatch_cfg.threads_w)));
    auto block_y = host_builder.create<mlir::arith::ConstantIndexOp>(
        loc, static_cast<int64_t>(std::max<uint32_t>(1u, dispatch_cfg.threads_h)));
    auto block_z = host_builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::gpu::KernelDim3 grid_size{grid_x, grid_y, grid_z};
    mlir::gpu::KernelDim3 block_size{block_x, block_y, block_z};
    host_builder.create<mlir::gpu::LaunchFuncOp>(loc,
                                                 gpu_func,
                                                 grid_size,
                                                 block_size,
                                                 mlir::Value{},
                                                 mlir::ValueRange{func.getArgument(0),
                                                                  func.getArgument(1),
                                                                  func.getArgument(2)},
                                                 mlir::Type{},
                                                 mlir::ValueRange{},
                                                 std::nullopt);
    host_builder.create<mlir::func::ReturnOp>(loc);
    return module;
}

}  // namespace

mlir::ModuleOp build_mlir_conv2d_vulkan(const std::shared_ptr<const ov::op::v1::Convolution>& conv,
                                        mlir::MLIRContext& ctx,
                                        const ParallelDispatchConfig* dispatch_cfg) {
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
                    mlir::gpu::GPUDialect,
                    mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect,
                    mlir::arith::ArithDialect,
                    mlir::math::MathDialect>();

    auto input_ty = to_memref_type(in_pshape, conv->get_input_element_type(0), ctx);
    auto weight_ty = to_memref_type(w_pshape, conv->get_input_element_type(1), ctx);
    auto output_ty = to_memref_type(out_pshape, conv->get_output_element_type(0), ctx);

    const auto& out_shape = conv->get_output_shape(0);
    const bool can_use_parallel_gpu_builder =
        dispatch_cfg != nullptr &&
        dispatch_cfg->enabled &&
        dispatch_cfg->threads_h > 0 &&
        dispatch_cfg->threads_w > 0 &&
        !out_shape.empty() &&
        out_shape[0] == 1;
    if (can_use_parallel_gpu_builder) {
        return build_parallel_conv2d_vulkan(conv, ctx, input_ty, weight_ty, output_ty, *dispatch_cfg);
    }
    return build_serial_conv2d_vulkan(conv, ctx, input_ty, weight_ty, output_ty);
}

}  // namespace gfx_plugin
}  // namespace ov
