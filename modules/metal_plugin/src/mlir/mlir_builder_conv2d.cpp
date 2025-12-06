// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_builder.hpp"

#include "openvino/op/convolution.hpp"
#include "openvino/core/model.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace ov {
namespace metal_plugin {

mlir::ModuleOp build_mlir_conv2d_from_model(const std::shared_ptr<const ov::Model>& model,
                                            mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::scf::SCFDialect, mlir::memref::MemRefDialect,
                    mlir::arith::ArithDialect>();
    std::shared_ptr<const ov::op::v1::Convolution> conv;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto c = ov::as_type_ptr<const ov::op::v1::Convolution>(node)) {
            conv = c;
            break;
        }
    }
    OPENVINO_ASSERT(conv, "Conv2D builder: Convolution op not found");
    auto shape = conv->get_input_shape(0);
    auto wshape = conv->get_input_shape(1);
    OPENVINO_ASSERT(shape.size() == 4 && wshape.size() == 4, "Conv2D builder expects rank-4 input/weights");

    const auto pads_begin = conv->get_pads_begin();  // {top, left}
    const auto strides    = conv->get_strides();
    const auto dilations  = conv->get_dilations();
    const uint32_t outH = conv->get_output_shape(0)[2];
    const uint32_t outW = conv->get_output_shape(0)[3];

    auto f32 = mlir::Float32Type::get(&ctx);
    mlir::SmallVector<int64_t> in_dims(shape.begin(), shape.end());
    mlir::SmallVector<int64_t> w_dims(wshape.begin(), wshape.end());
    mlir::SmallVector<int64_t> out_dims(conv->get_output_shape(0).begin(), conv->get_output_shape(0).end());

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());
    auto func_type = mb.getFunctionType({mlir::MemRefType::get(in_dims, f32),
                                         mlir::MemRefType::get(w_dims, f32)},
                                        {mlir::MemRefType::get(out_dims, f32)});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "conv2d_main", func_type);
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

    auto N = make_idx(shape[0]);
    auto C_in = make_idx(shape[1]);
    auto H = make_idx(shape[2]);
    auto W = make_idx(shape[3]);
    auto C_out = make_idx(wshape[0]);
    auto kH = make_idx(wshape[2]);
    auto kW = make_idx(wshape[3]);
    auto outH_c = make_idx(outH);
    auto outW_c = make_idx(outW);

    auto strideH = make_idx(strides[0]);
    auto strideW = make_idx(strides[1]);
    auto dilH = make_idx(dilations[0]);
    auto dilW = make_idx(dilations[1]);
    auto padTop = make_idx(pads_begin[0]);
    auto padLeft = make_idx(pads_begin[1]);

    auto for_n = b.create<mlir::scf::ForOp>(loc, c0, N, c1);
    auto bn = mlir::OpBuilder::atBlockBegin(for_n.getBody());
    auto for_oc = bn.create<mlir::scf::ForOp>(loc, c0, C_out, c1);
    auto boc = mlir::OpBuilder::atBlockBegin(for_oc.getBody());
    auto for_oh = boc.create<mlir::scf::ForOp>(loc, c0, outH_c, c1);
    auto boh = mlir::OpBuilder::atBlockBegin(for_oh.getBody());
    auto for_ow = boh.create<mlir::scf::ForOp>(loc, c0, outW_c, c1);
    auto bow = mlir::OpBuilder::atBlockBegin(for_ow.getBody());

    auto acc = bow.create<mlir::arith::ConstantOp>(loc, b.getF32FloatAttr(0.0f));
    auto for_ic = bow.create<mlir::scf::ForOp>(loc, c0, C_in, c1, acc.getResult());
    auto bic = mlir::OpBuilder::atBlockBegin(for_ic.getBody());
    auto for_kh = bic.create<mlir::scf::ForOp>(loc, c0, kH, c1, for_ic.getRegionIterArgs()[0]);
    auto bkh = mlir::OpBuilder::atBlockBegin(for_kh.getBody());
    auto for_kw = bkh.create<mlir::scf::ForOp>(loc, c0, kW, c1, for_kh.getRegionIterArgs()[0]);
    auto bkw = mlir::OpBuilder::atBlockBegin(for_kw.getBody());

    auto oh_mul_stride = bkw.create<mlir::arith::MulIOp>(loc, for_oh.getInductionVar(), strideH);
    auto ih_tmp = bkw.create<mlir::arith::SubIOp>(loc, oh_mul_stride, padTop);
    auto kh_dil = bkw.create<mlir::arith::MulIOp>(loc, for_kh.getInductionVar(), dilH);
    auto ih = bkw.create<mlir::arith::AddIOp>(loc, ih_tmp, kh_dil);

    auto ow_mul_stride = bkw.create<mlir::arith::MulIOp>(loc, for_ow.getInductionVar(), strideW);
    auto iw_tmp = bkw.create<mlir::arith::SubIOp>(loc, ow_mul_stride, padLeft);
    auto kw_dil = bkw.create<mlir::arith::MulIOp>(loc, for_kw.getInductionVar(), dilW);
    auto iw = bkw.create<mlir::arith::AddIOp>(loc, iw_tmp, kw_dil);

    auto inp = bkw.create<mlir::memref::LoadOp>(loc, func.getArgument(0),
                                                mlir::ValueRange{for_n.getInductionVar(),
                                                                 for_ic.getInductionVar(),
                                                                 ih,
                                                                 iw});
    auto wgt = bkw.create<mlir::memref::LoadOp>(loc, func.getArgument(1),
                                                mlir::ValueRange{for_oc.getInductionVar(),
                                                                 for_ic.getInductionVar(),
                                                                 for_kh.getInductionVar(),
                                                                 for_kw.getInductionVar()});
    auto mul = bkw.create<mlir::arith::MulFOp>(loc, inp, wgt);
    auto add = bkw.create<mlir::arith::AddFOp>(loc, for_kw.getRegionIterArgs()[0], mul);
    bkw.create<mlir::scf::YieldOp>(loc, add.getResult());

    bow.create<mlir::memref::StoreOp>(loc, for_ic.getResult(0), out_alloc,
                                      mlir::ValueRange{for_n.getInductionVar(),
                                                       for_oc.getInductionVar(),
                                                       for_oh.getInductionVar(),
                                                       for_ow.getInductionVar()});
    bow.create<mlir::scf::YieldOp>(loc);
    boh.create<mlir::scf::YieldOp>(loc);
    boc.create<mlir::scf::YieldOp>(loc);
    bn.create<mlir::scf::YieldOp>(loc);

    b.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{out_alloc});
    return module;
}

}  // namespace metal_plugin
}  // namespace ov
