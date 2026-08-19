// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "openvino/op/avg_pool.hpp"
#include "openvino/core/model.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace ov {
namespace gfx_plugin {

mlir::ModuleOp build_mlir_avgpool_from_model(const std::shared_ptr<const ov::Model>& model,
                                             mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::scf::SCFDialect, mlir::memref::MemRefDialect,
                    mlir::arith::ArithDialect>();
    std::shared_ptr<const ov::op::v1::AvgPool> pool;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto p = ov::as_type_ptr<const ov::op::v1::AvgPool>(node)) { pool = p; break; }
    }
    OPENVINO_ASSERT(pool, "AvgPool builder: AvgPool op not found");

    const auto in_shape = pool->get_input_shape(0);
    const auto out_shape = pool->get_output_shape(0);
    const auto strides = pool->get_strides();
    const auto pads_begin = pool->get_pads_begin();
    const auto kernel = pool->get_kernel();

    auto to_elem_ty = [&](ov::element::Type et) -> mlir::Type {
        switch (et) {
            case ov::element::f16: return mlir::Float16Type::get(&ctx);
            case ov::element::f32: return mlir::Float32Type::get(&ctx);
            default: return mlir::Float32Type::get(&ctx);
        }
    };
    auto elem_ty = to_elem_ty(pool->get_input_element_type(0));
    mlir::SmallVector<int64_t> in_dims(in_shape.begin(), in_shape.end());
    mlir::SmallVector<int64_t> out_dims(out_shape.begin(), out_shape.end());

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());
    auto func_type = mb.getFunctionType({mlir::MemRefType::get(in_dims, elem_ty)},
                                        {mlir::MemRefType::get(out_dims, elem_ty)});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "avgpool_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    b.setInsertionPointToStart(&func.getBody().front());
    auto loc = mlir::UnknownLoc::get(&ctx);
    auto c0 = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1 = b.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto out_alloc = b.create<mlir::memref::AllocOp>(
        loc,
        llvm::cast<mlir::MemRefType>(func.getResultTypes().front()));
    const int64_t thread_h_value = 8;
    const int64_t thread_w_value = 8;
    const int64_t tile_h_value = thread_h_value;
    const int64_t tile_w_value = thread_w_value;
    module->setAttr("gfx.dispatch_tile_h",
                    mlir::IntegerAttr::get(mlir::IndexType::get(&ctx), tile_h_value));
    module->setAttr("gfx.dispatch_tile_w",
                    mlir::IntegerAttr::get(mlir::IndexType::get(&ctx), tile_w_value));
    module->setAttr("gfx.dispatch_threads_h",
                    mlir::IntegerAttr::get(mlir::IndexType::get(&ctx), thread_h_value));
    module->setAttr("gfx.dispatch_threads_w",
                    mlir::IntegerAttr::get(mlir::IndexType::get(&ctx), thread_w_value));
    module->setAttr("gfx.parallel_loop_dims",
                    mlir::IntegerAttr::get(mlir::IntegerType::get(&ctx, 64), 5));

    auto make_idx = [&](int64_t v) { return b.create<mlir::arith::ConstantIndexOp>(loc, v); };
    auto N = make_idx(in_shape[0]);
    auto C = make_idx(in_shape[1]);
    auto H = make_idx(in_shape[2]);
    auto W = make_idx(in_shape[3]);
    auto outH = make_idx(out_shape[2]);
    auto outW = make_idx(out_shape[3]);
    auto kH = make_idx(kernel[0]);
    auto kW = make_idx(kernel[1]);
    auto strideH = make_idx(strides[0]);
    auto strideW = make_idx(strides[1]);
    auto dilH = make_idx(1);
    auto dilW = make_idx(1);
    auto padTop = make_idx(pads_begin[0]);
    auto padLeft = make_idx(pads_begin[1]);
    auto tileH = make_idx(tile_h_value);
    auto tileW = make_idx(tile_w_value);
    auto threadH = make_idx(thread_h_value);
    auto threadW = make_idx(thread_w_value);
    auto tileHMinus1 = make_idx(tile_h_value - 1);
    auto tileWMinus1 = make_idx(tile_w_value - 1);
    const bool exclude_pad = pool->get_exclude_pad();
    auto kernel_area = b.create<mlir::arith::ConstantOp>(
        loc,
        mlir::FloatAttr::get(elem_ty, static_cast<float>(kernel[0] * kernel[1])));
    auto zero = b.create<mlir::arith::ConstantOp>(loc, mlir::FloatAttr::get(elem_ty, 0.0f));
    auto one = b.create<mlir::arith::ConstantOp>(loc, mlir::FloatAttr::get(elem_ty, 1.0f));

    auto hTilesNum = b.create<mlir::arith::AddIOp>(loc, outH, tileHMinus1);
    auto wTilesNum = b.create<mlir::arith::AddIOp>(loc, outW, tileWMinus1);
    auto HTiles = b.create<mlir::arith::DivSIOp>(loc, hTilesNum, tileH);
    auto WTiles = b.create<mlir::arith::DivSIOp>(loc, wTilesNum, tileW);

    auto par = b.create<mlir::scf::ParallelOp>(
        loc,
        mlir::ValueRange{c0, c0, c0, c0, c0},
        mlir::ValueRange{C, HTiles, WTiles, threadH, threadW},
        mlir::ValueRange{c1, c1, c1, c1, c1});
    b.setInsertionPoint(par.getBody()->getTerminator());
    auto ivs = par.getInductionVars();
    auto iv_c = ivs[0];
    auto oh_base = b.create<mlir::arith::MulIOp>(loc, ivs[1], tileH);
    auto ow_base = b.create<mlir::arith::MulIOp>(loc, ivs[2], tileW);
    auto oh = b.create<mlir::arith::AddIOp>(loc, oh_base, ivs[3]);
    auto ow = b.create<mlir::arith::AddIOp>(loc, ow_base, ivs[4]);
    auto oh_in = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, oh, outH);
    auto ow_in = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, ow, outW);
    auto out_in = b.create<mlir::arith::AndIOp>(loc, oh_in, ow_in);
    auto if_out = b.create<mlir::scf::IfOp>(loc, out_in, /*withElse=*/false);
    {
        mlir::OpBuilder::InsertionGuard if_guard(b);
        b.setInsertionPointToStart(&if_out.getThenRegion().front());
        auto for_n = b.create<mlir::scf::ForOp>(
            loc, c0, N, c1, mlir::ValueRange{},
            [&](mlir::OpBuilder& bn, mlir::Location body_loc, mlir::Value iv_n, mlir::ValueRange) {
                auto for_kh = bn.create<mlir::scf::ForOp>(
                    body_loc, c0, kH, c1, mlir::ValueRange{zero.getResult(), zero.getResult()},
                    [&](mlir::OpBuilder& bkh, mlir::Location kh_loc, mlir::Value iv_kh, mlir::ValueRange kh_args) {
                        auto for_kw = bkh.create<mlir::scf::ForOp>(
                            kh_loc, c0, kW, c1, mlir::ValueRange{kh_args[0], kh_args[1]},
                            [&](mlir::OpBuilder& bkw, mlir::Location kw_loc, mlir::Value iv_kw, mlir::ValueRange kw_args) {
                                auto oh_stride = bkw.create<mlir::arith::MulIOp>(kw_loc, oh, strideH);
                                auto ih_tmp = bkw.create<mlir::arith::SubIOp>(kw_loc, oh_stride, padTop);
                                auto kh_d = bkw.create<mlir::arith::MulIOp>(kw_loc, iv_kh, dilH);
                                auto ih = bkw.create<mlir::arith::AddIOp>(kw_loc, ih_tmp, kh_d);
                                auto ow_stride = bkw.create<mlir::arith::MulIOp>(kw_loc, ow, strideW);
                                auto iw_tmp = bkw.create<mlir::arith::SubIOp>(kw_loc, ow_stride, padLeft);
                                auto kw_d = bkw.create<mlir::arith::MulIOp>(kw_loc, iv_kw, dilW);
                                auto iw = bkw.create<mlir::arith::AddIOp>(kw_loc, iw_tmp, kw_d);
                                auto ih_ge = bkw.create<mlir::arith::CmpIOp>(
                                    kw_loc, mlir::arith::CmpIPredicate::sge, ih, c0);
                                auto ih_lt = bkw.create<mlir::arith::CmpIOp>(
                                    kw_loc, mlir::arith::CmpIPredicate::slt, ih, H);
                                auto iw_ge = bkw.create<mlir::arith::CmpIOp>(
                                    kw_loc, mlir::arith::CmpIPredicate::sge, iw, c0);
                                auto iw_lt = bkw.create<mlir::arith::CmpIOp>(
                                    kw_loc, mlir::arith::CmpIPredicate::slt, iw, W);
                                auto in_h = bkw.create<mlir::arith::AndIOp>(kw_loc, ih_ge, ih_lt);
                                auto in_w = bkw.create<mlir::arith::AndIOp>(kw_loc, iw_ge, iw_lt);
                                auto in_bounds = bkw.create<mlir::arith::AndIOp>(kw_loc, in_h, in_w);
                                auto if_in = bkw.create<mlir::scf::IfOp>(
                                    kw_loc,
                                    mlir::TypeRange{kw_args[0].getType(), kw_args[1].getType()},
                                    in_bounds,
                                    /*withElse=*/true);
                                {
                                    mlir::OpBuilder::InsertionGuard then_guard(bkw);
                                    bkw.setInsertionPointToStart(&if_in.getThenRegion().front());
                                    auto inp = bkw.create<mlir::memref::LoadOp>(
                                        kw_loc, func.getArgument(0), mlir::ValueRange{iv_n, iv_c, ih, iw});
                                    auto sum = bkw.create<mlir::arith::AddFOp>(kw_loc, kw_args[0], inp);
                                    auto count = bkw.create<mlir::arith::AddFOp>(kw_loc, kw_args[1], one.getResult());
                                    bkw.create<mlir::scf::YieldOp>(
                                        kw_loc, mlir::ValueRange{sum.getResult(), count.getResult()});
                                }
                                {
                                    mlir::OpBuilder::InsertionGuard else_guard(bkw);
                                    bkw.setInsertionPointToStart(&if_in.getElseRegion().front());
                                    bkw.create<mlir::scf::YieldOp>(
                                        kw_loc, mlir::ValueRange{kw_args[0], kw_args[1]});
                                }
                                bkw.create<mlir::scf::YieldOp>(
                                    kw_loc, mlir::ValueRange{if_in.getResult(0), if_in.getResult(1)});
                            });
                        bkh.create<mlir::scf::YieldOp>(
                            kh_loc, mlir::ValueRange{for_kw.getResult(0), for_kw.getResult(1)});
                    });
                auto denom = exclude_pad ? for_kh.getResult(1) : kernel_area.getResult();
                auto avg = bn.create<mlir::arith::DivFOp>(body_loc, for_kh.getResult(0), denom);
                bn.create<mlir::memref::StoreOp>(
                    body_loc, avg.getResult(), out_alloc, mlir::ValueRange{iv_n, iv_c, oh, ow});
                bn.create<mlir::scf::YieldOp>(body_loc);
            });
        (void)for_n;
    }

    b.setInsertionPointAfter(par);
    b.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{out_alloc});
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
