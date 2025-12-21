// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_builder.hpp"

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

    auto f32 = mlir::Float32Type::get(&ctx);
    mlir::SmallVector<int64_t> in_dims(in_shape.begin(), in_shape.end());
    mlir::SmallVector<int64_t> out_dims(out_shape.begin(), out_shape.end());

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());
    auto func_type = mb.getFunctionType({mlir::MemRefType::get(in_dims, f32)},
                                        {mlir::MemRefType::get(out_dims, f32)});
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

    auto for_n = b.create<mlir::scf::ForOp>(loc, c0, N, c1);
    auto bn = mlir::OpBuilder::atBlockBegin(for_n.getBody());
    auto for_c = bn.create<mlir::scf::ForOp>(loc, c0, C, c1);
    auto bc = mlir::OpBuilder::atBlockBegin(for_c.getBody());
    auto for_oh = bc.create<mlir::scf::ForOp>(loc, c0, outH, c1);
    auto boh = mlir::OpBuilder::atBlockBegin(for_oh.getBody());
    auto for_ow = boh.create<mlir::scf::ForOp>(loc, c0, outW, c1);
    auto bow = mlir::OpBuilder::atBlockBegin(for_ow.getBody());
    auto acc = bow.create<mlir::arith::ConstantOp>(loc, b.getF32FloatAttr(0.0f));
    auto for_kh = bow.create<mlir::scf::ForOp>(loc, c0, kH, c1, acc.getResult());
    auto bkh = mlir::OpBuilder::atBlockBegin(for_kh.getBody());
    auto for_kw = bkh.create<mlir::scf::ForOp>(loc, c0, kW, c1, for_kh.getRegionIterArgs()[0]);
    auto bkw = mlir::OpBuilder::atBlockBegin(for_kw.getBody());

    auto oh_stride = bkw.create<mlir::arith::MulIOp>(loc, for_oh.getInductionVar(), strideH);
    auto ih_tmp = bkw.create<mlir::arith::SubIOp>(loc, oh_stride, padTop);
    auto kh_d = bkw.create<mlir::arith::MulIOp>(loc, for_kh.getInductionVar(), dilH);
    auto ih = bkw.create<mlir::arith::AddIOp>(loc, ih_tmp, kh_d);

    auto ow_stride = bkw.create<mlir::arith::MulIOp>(loc, for_ow.getInductionVar(), strideW);
    auto iw_tmp = bkw.create<mlir::arith::SubIOp>(loc, ow_stride, padLeft);
    auto kw_d = bkw.create<mlir::arith::MulIOp>(loc, for_kw.getInductionVar(), dilW);
    auto iw = bkw.create<mlir::arith::AddIOp>(loc, iw_tmp, kw_d);

    auto inp = bkw.create<mlir::memref::LoadOp>(loc, func.getArgument(0),
                                                mlir::ValueRange{for_n.getInductionVar(),
                                                                 for_c.getInductionVar(),
                                                                 ih,
                                                                 iw});
    auto sum = bkw.create<mlir::arith::AddFOp>(loc, for_kw.getRegionIterArgs()[0], inp);
    bkw.create<mlir::scf::YieldOp>(loc, sum.getResult());

    auto avg = bow.create<mlir::arith::DivFOp>(
        loc,
        for_kh.getResult(0),
        b.create<mlir::arith::ConstantOp>(loc, b.getF32FloatAttr(static_cast<float>(kernel[0] * kernel[1]))));
    bow.create<mlir::memref::StoreOp>(loc, avg.getResult(), out_alloc,
                                      mlir::ValueRange{for_n.getInductionVar(),
                                                       for_c.getInductionVar(),
                                                       for_oh.getInductionVar(),
                                                       for_ow.getInductionVar()});
    bow.create<mlir::scf::YieldOp>(loc);
    boh.create<mlir::scf::YieldOp>(loc);
    bc.create<mlir::scf::YieldOp>(loc);
    bn.create<mlir::scf::YieldOp>(loc);

    b.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{out_alloc});
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov

