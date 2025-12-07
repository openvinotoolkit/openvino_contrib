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
#include "mlir/IR/BuiltinTypes.h"

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
    const auto in_pshape = conv->get_input_partial_shape(0);
    const auto w_pshape = conv->get_input_partial_shape(1);
    const auto out_pshape = conv->get_output_partial_shape(0);
    OPENVINO_ASSERT(in_pshape.rank().is_static() && in_pshape.rank().get_length() == 4,
                    "Conv2D builder expects rank-4 input");
    OPENVINO_ASSERT(w_pshape.rank().is_static() && w_pshape.rank().get_length() == 4,
                    "Conv2D builder expects rank-4 weights");
    OPENVINO_ASSERT(out_pshape.rank().is_static() && out_pshape.rank().get_length() == 4,
                    "Conv2D builder expects rank-4 output");

    const auto pads_begin = conv->get_pads_begin();  // {top, left}
    const auto pads_end   = conv->get_pads_end();    // {bottom, right}
    const auto strides    = conv->get_strides();
    const auto dilations  = conv->get_dilations();

    auto f32 = mlir::Float32Type::get(&ctx);

    auto to_memref_shape = [](const ov::PartialShape& ps) {
        mlir::SmallVector<int64_t> dims;
        dims.reserve(ps.rank().get_length());
        for (const auto& d : ps) {
            dims.push_back(d.is_dynamic() ? mlir::ShapedType::kDynamic
                                          : static_cast<int64_t>(d.get_length()));
        }
        return dims;
    };

    const auto in_dims  = to_memref_shape(in_pshape);
    const auto w_dims   = to_memref_shape(w_pshape);
    const auto out_dims = to_memref_shape(out_pshape);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto in_type  = mlir::MemRefType::get(in_dims, f32);
    auto w_type   = mlir::MemRefType::get(w_dims, f32);
    auto out_type = mlir::MemRefType::get(out_dims, f32);

    auto func_type = mb.getFunctionType({in_type, w_type}, {out_type});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "conv2d_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    b.setInsertionPointToStart(&func.getBody().front());
    auto loc = mlir::UnknownLoc::get(&ctx);

    auto c0  = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1  = b.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto one = b.create<mlir::arith::ConstantIndexOp>(loc, 1);

    auto input  = func.getArgument(0);
    auto weight = func.getArgument(1);

    auto N     = b.create<mlir::memref::DimOp>(loc, input, 0);
    auto C_in  = b.create<mlir::memref::DimOp>(loc, input, 1);
    auto H     = b.create<mlir::memref::DimOp>(loc, input, 2);
    auto W     = b.create<mlir::memref::DimOp>(loc, input, 3);
    auto C_out = b.create<mlir::memref::DimOp>(loc, weight, 0);
    auto kH    = b.create<mlir::memref::DimOp>(loc, weight, 2);
    auto kW    = b.create<mlir::memref::DimOp>(loc, weight, 3);

    auto padTop    = b.create<mlir::arith::ConstantIndexOp>(loc, pads_begin[0]);
    auto padLeft   = b.create<mlir::arith::ConstantIndexOp>(loc, pads_begin[1]);
    auto padBottom = b.create<mlir::arith::ConstantIndexOp>(loc, pads_end[0]);
    auto padRight  = b.create<mlir::arith::ConstantIndexOp>(loc, pads_end[1]);
    auto strideH   = b.create<mlir::arith::ConstantIndexOp>(loc, strides[0]);
    auto strideW   = b.create<mlir::arith::ConstantIndexOp>(loc, strides[1]);
    auto dilH      = b.create<mlir::arith::ConstantIndexOp>(loc, dilations[0]);
    auto dilW      = b.create<mlir::arith::ConstantIndexOp>(loc, dilations[1]);

    auto kh_minus_1  = b.create<mlir::arith::SubIOp>(loc, kH, one);
    auto kw_minus_1  = b.create<mlir::arith::SubIOp>(loc, kW, one);
    auto eff_filterH = b.create<mlir::arith::MulIOp>(loc, dilH, kh_minus_1);
    auto eff_filterW = b.create<mlir::arith::MulIOp>(loc, dilW, kw_minus_1);

    auto H_pad   = b.create<mlir::arith::AddIOp>(loc, H, padTop);
    auto H_pad2  = b.create<mlir::arith::AddIOp>(loc, H_pad, padBottom);
    auto H_sub   = b.create<mlir::arith::SubIOp>(loc, H_pad2, eff_filterH);
    auto H_sub1  = b.create<mlir::arith::SubIOp>(loc, H_sub, one);
    auto outHdiv = b.create<mlir::arith::DivSIOp>(loc, H_sub1, strideH);
    auto outH    = b.create<mlir::arith::AddIOp>(loc, outHdiv, one);

    auto W_pad   = b.create<mlir::arith::AddIOp>(loc, W, padLeft);
    auto W_pad2  = b.create<mlir::arith::AddIOp>(loc, W_pad, padRight);
    auto W_sub   = b.create<mlir::arith::SubIOp>(loc, W_pad2, eff_filterW);
    auto W_sub1  = b.create<mlir::arith::SubIOp>(loc, W_sub, one);
    auto outWdiv = b.create<mlir::arith::DivSIOp>(loc, W_sub1, strideW);
    auto outW    = b.create<mlir::arith::AddIOp>(loc, outWdiv, one);

    mlir::SmallVector<mlir::Value> out_dyn_sizes;
    const auto out_shape = out_type.getShape();
    auto push_if_dynamic = [&](int idx, mlir::Value v) {
        if (out_shape[idx] == mlir::ShapedType::kDynamic)
            out_dyn_sizes.push_back(v);
    };
    push_if_dynamic(0, N);
    push_if_dynamic(1, C_out);
    push_if_dynamic(2, outH);
    push_if_dynamic(3, outW);

    auto out_alloc = b.create<mlir::memref::AllocOp>(loc, out_type, out_dyn_sizes);

    auto for_n  = b.create<mlir::scf::ForOp>(loc, c0, N, c1);
    auto bn     = mlir::OpBuilder::atBlockBegin(for_n.getBody());
    auto for_oc = bn.create<mlir::scf::ForOp>(loc, c0, C_out, c1);
    auto boc    = mlir::OpBuilder::atBlockBegin(for_oc.getBody());
    auto for_oh = boc.create<mlir::scf::ForOp>(loc, c0, outH, c1);
    auto boh    = mlir::OpBuilder::atBlockBegin(for_oh.getBody());
    auto for_ow = boh.create<mlir::scf::ForOp>(loc, c0, outW, c1);
    auto bow    = mlir::OpBuilder::atBlockBegin(for_ow.getBody());

    auto acc    = bow.create<mlir::arith::ConstantOp>(loc, b.getF32FloatAttr(0.0f));
    auto for_ic = bow.create<mlir::scf::ForOp>(loc, c0, C_in, c1, acc.getResult());
    auto bic    = mlir::OpBuilder::atBlockBegin(for_ic.getBody());
    auto for_kh = bic.create<mlir::scf::ForOp>(loc, c0, kH, c1, for_ic.getRegionIterArgs()[0]);
    auto bkh    = mlir::OpBuilder::atBlockBegin(for_kh.getBody());
    auto for_kw = bkh.create<mlir::scf::ForOp>(loc, c0, kW, c1, for_kh.getRegionIterArgs()[0]);
    auto bkw    = mlir::OpBuilder::atBlockBegin(for_kw.getBody());

    auto oh_mul_stride = bkw.create<mlir::arith::MulIOp>(loc, for_oh.getInductionVar(), strideH);
    auto ih_tmp        = bkw.create<mlir::arith::SubIOp>(loc, oh_mul_stride, padTop);
    auto kh_dil        = bkw.create<mlir::arith::MulIOp>(loc, for_kh.getInductionVar(), dilH);
    auto ih            = bkw.create<mlir::arith::AddIOp>(loc, ih_tmp, kh_dil);

    auto ow_mul_stride = bkw.create<mlir::arith::MulIOp>(loc, for_ow.getInductionVar(), strideW);
    auto iw_tmp        = bkw.create<mlir::arith::SubIOp>(loc, ow_mul_stride, padLeft);
    auto kw_dil        = bkw.create<mlir::arith::MulIOp>(loc, for_kw.getInductionVar(), dilW);
    auto iw            = bkw.create<mlir::arith::AddIOp>(loc, iw_tmp, kw_dil);

    auto inp = bkw.create<mlir::memref::LoadOp>(loc, input,
                                                mlir::ValueRange{for_n.getInductionVar(),
                                                                 for_ic.getInductionVar(),
                                                                 ih,
                                                                 iw});
    auto wgt = bkw.create<mlir::memref::LoadOp>(loc, weight,
                                                mlir::ValueRange{for_oc.getInductionVar(),
                                                                 for_ic.getInductionVar(),
                                                                 for_kh.getInductionVar(),
                                                                 for_kw.getInductionVar()});
    auto mul = bkw.create<mlir::arith::MulFOp>(loc, inp, wgt);
    auto add = bkw.create<mlir::arith::AddFOp>(loc, for_kw.getRegionIterArgs()[0], mul);
    bkw.create<mlir::scf::YieldOp>(loc, add.getResult());

    bkh.create<mlir::scf::YieldOp>(loc, for_kw.getResult(0));
    bic.create<mlir::scf::YieldOp>(loc, for_kh.getResult(0));

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
