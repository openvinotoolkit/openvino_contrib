// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transforms/conv_parallel_lowering.hpp"

#include <stdexcept>

#include "runtime/gfx_logger.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

namespace ov {
namespace gfx_plugin {

namespace {

bool extract_hw(mlir::DenseIntElementsAttr attr, int64_t& h, int64_t& w) {
    if (!attr || attr.getNumElements() != 2) {
        return false;
    }
    auto it = attr.getValues<int64_t>().begin();
    h = *it++;
    w = *it++;
    return true;
}

bool extract_addi_offset(mlir::Value value, int64_t& offset) {
    if (auto add = value.getDefiningOp<mlir::arith::AddIOp>()) {
        if (auto cst = add.getLhs().getDefiningOp<mlir::arith::ConstantIndexOp>()) {
            offset = cst.value();
            return true;
        }
        if (auto cst = add.getRhs().getDefiningOp<mlir::arith::ConstantIndexOp>()) {
            offset = cst.value();
            return true;
        }
    }
    return false;
}

mlir::Value strip_memref_casts(mlir::Value value) {
    mlir::Value current = value;
    while (current) {
        if (auto cast = current.getDefiningOp<mlir::memref::CastOp>()) {
            current = cast.getSource();
            continue;
        }
        if (auto subview = current.getDefiningOp<mlir::memref::SubViewOp>()) {
            current = subview.getSource();
            continue;
        }
        if (auto view = current.getDefiningOp<mlir::memref::ViewOp>()) {
            current = view.getSource();
            continue;
        }
        break;
    }
    return current;
}

bool lower_conv2d_op(mlir::linalg::Conv2DNchwFchwOp op, mlir::IRRewriter& rewriter) {
    if (op.getInputs().size() < 2 || op.getOutputs().empty()) {
        return false;
    }

    mlir::Value input = op.getInputs()[0];
    mlir::Value filter = op.getInputs()[1];
    mlir::Value output = op.getOutputs()[0];

    auto in_type = mlir::dyn_cast<mlir::MemRefType>(input.getType());
    auto w_type = mlir::dyn_cast<mlir::MemRefType>(filter.getType());
    auto out_type = mlir::dyn_cast<mlir::MemRefType>(output.getType());
    if (!in_type || !w_type || !out_type) {
        return false;  // not bufferized
    }
    if (in_type.getRank() != 4 || w_type.getRank() != 4 || out_type.getRank() != 4) {
        return false;
    }

    auto elem_ty = out_type.getElementType();
    if (!mlir::isa<mlir::FloatType>(elem_ty)) {
        return false;
    }

    mlir::Value conv_input = input;
    mlir::Value conv_output = output;
    mlir::Value input_base = strip_memref_casts(input);
    mlir::Value output_base = strip_memref_casts(output);
    int64_t pad_h = 0;
    int64_t pad_w = 0;
    int64_t pad_end_h = 0;
    int64_t pad_end_w = 0;
    mlir::scf::ParallelOp pad_fill_parallel;
    if (auto pad_attr = op->getAttrOfType<mlir::DenseIntElementsAttr>("gfx.pad_begin")) {
        (void)extract_hw(pad_attr, pad_h, pad_w);
    }
    if (auto pad_attr = op->getAttrOfType<mlir::DenseIntElementsAttr>("gfx.pad_end")) {
        (void)extract_hw(pad_attr, pad_end_h, pad_end_w);
    }

    bool found_copy = false;
    if (auto func = op->getParentOfType<mlir::func::FuncOp>()) {
        func.walk([&](mlir::memref::StoreOp store) {
            if (strip_memref_casts(store.getMemRef()) != input_base) {
                return;
            }
            if (!found_copy) {
                if (auto load = store.getValue().getDefiningOp<mlir::memref::LoadOp>()) {
                    conv_input = strip_memref_casts(load.getMemRef());
                    auto indices = store.getIndices();
                    if (indices.size() >= 4) {
                        (void)extract_addi_offset(indices[2], pad_h);
                        (void)extract_addi_offset(indices[3], pad_w);
                    }
                    found_copy = true;
                    return;
                }
            }
            if (!pad_fill_parallel) {
                if (auto cst = store.getValue().getDefiningOp<mlir::arith::ConstantOp>()) {
                    if (auto fattr = mlir::dyn_cast<mlir::FloatAttr>(cst.getValue())) {
                        if (fattr.getValueAsDouble() == 0.0) {
                            pad_fill_parallel = store->getParentOfType<mlir::scf::ParallelOp>();
                        }
                    }
                }
            }
        });
    }

    if (auto func = op->getParentOfType<mlir::func::FuncOp>()) {
        auto padded_type = mlir::dyn_cast<mlir::MemRefType>(input_base.getType());
        if (padded_type && padded_type.getRank() == 4) {
            auto padded_shape = padded_type.getShape();
            if (padded_shape[2] != mlir::ShapedType::kDynamic &&
                padded_shape[3] != mlir::ShapedType::kDynamic) {
                const int64_t orig_h = padded_shape[2] - pad_h - pad_end_h;
                const int64_t orig_w = padded_shape[3] - pad_w - pad_end_w;
                for (auto arg : func.getArguments()) {
                    auto arg_type = mlir::dyn_cast<mlir::MemRefType>(arg.getType());
                    if (!arg_type || arg_type.getRank() != 4) {
                        continue;
                    }
                    if (arg_type.getElementType() != padded_type.getElementType()) {
                        continue;
                    }
                    auto arg_shape = arg_type.getShape();
                    if (arg_shape[0] != padded_shape[0] || arg_shape[1] != padded_shape[1]) {
                        continue;
                    }
                    if (arg_shape[2] == orig_h && arg_shape[3] == orig_w) {
                        conv_input = strip_memref_casts(arg);
                        break;
                    }
                }
            }
        }
    }

    const bool using_padded_input = (strip_memref_casts(conv_input) == input_base);
    if (gfx_log_debug_enabled()) {
        GFX_LOG_DEBUG("MLIR", "Conv2D pad detect: conv_input="
                                  << (using_padded_input ? "padded" : "orig")
                                  << " pad_h=" << pad_h << " pad_w=" << pad_w
                                  << " pad_end_h=" << pad_end_h << " pad_end_w=" << pad_end_w
                                  << " conv_output="
                                  << (conv_output == output ? "alloc" : "arg"));
    }

    mlir::linalg::FillOp fill_op;
    mlir::Value zero_init;
    for (auto* user : output_base.getUsers()) {
        auto fill = mlir::dyn_cast<mlir::linalg::FillOp>(user);
        if (!fill || !fill->isBeforeInBlock(op)) {
            continue;
        }
        if (fill.getInputs().empty()) {
            continue;
        }
        auto cst = fill.getInputs()[0].getDefiningOp<mlir::arith::ConstantOp>();
        if (!cst) {
            continue;
        }
        if (auto fattr = mlir::dyn_cast<mlir::FloatAttr>(cst.getValue())) {
            if (fattr.getValueAsDouble() == 0.0) {
                fill_op = fill;
                zero_init = fill.getInputs()[0];
                break;
            }
        }
    }

    int64_t stride_h = 0, stride_w = 0;
    int64_t dil_h = 0, dil_w = 0;
    if (!extract_hw(op.getStrides(), stride_h, stride_w) ||
        !extract_hw(op.getDilations(), dil_h, dil_w)) {
        return false;
    }

    if (op->getNumResults() > 0) {
        if (op->getNumResults() != 1 || op->getResult(0).getType() != output.getType()) {
            return false;
        }
    }

    const auto loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    auto c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto strideH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, stride_h);
    auto strideW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, stride_w);
    auto dilH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, dil_h);
    auto dilW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, dil_w);

    auto N = rewriter.create<mlir::memref::DimOp>(loc, conv_output, 0);
    auto C_out = rewriter.create<mlir::memref::DimOp>(loc, conv_output, 1);
    auto H_out = rewriter.create<mlir::memref::DimOp>(loc, conv_output, 2);
    auto W_out = rewriter.create<mlir::memref::DimOp>(loc, conv_output, 3);
    auto C_in = rewriter.create<mlir::memref::DimOp>(loc, conv_input, 1);
    auto H_in = rewriter.create<mlir::memref::DimOp>(loc, conv_input, 2);
    auto W_in = rewriter.create<mlir::memref::DimOp>(loc, conv_input, 3);
    auto kH = rewriter.create<mlir::memref::DimOp>(loc, filter, 2);
    auto kW = rewriter.create<mlir::memref::DimOp>(loc, filter, 3);

    // Map parallel loops to output C/H/W so the Vulkan dispatch grid (last 3 dims)
    // matches block mapping: grid = [C_out, H_out, W_out].
    auto par = rewriter.create<mlir::scf::ParallelOp>(
        loc,
        mlir::ValueRange{c0, c0, c0},
        mlir::ValueRange{C_out, H_out, W_out},
        mlir::ValueRange{c1, c1, c1});

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(par.getBody()->getTerminator());

    auto ivs = par.getInductionVars();
    auto iv_oc = ivs[0];
    auto iv_oh = ivs[1];
    auto iv_ow = ivs[2];
    const bool needs_bounds = !using_padded_input;
    const int64_t effective_pad_h = using_padded_input ? 0 : pad_h;
    const int64_t effective_pad_w = using_padded_input ? 0 : pad_w;
    auto padH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, effective_pad_h);
    auto padW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, effective_pad_w);

    rewriter.create<mlir::scf::ForOp>(
        loc, c0, N, c1, mlir::ValueRange{},
        [&](mlir::OpBuilder& b, mlir::Location body_loc, mlir::Value iv_n, mlir::ValueRange) {
            mlir::Value acc0;
            if (zero_init) {
                acc0 = b.create<mlir::arith::ConstantOp>(
                    body_loc, elem_ty, b.getFloatAttr(elem_ty, 0.0));
            } else {
                acc0 = b.create<mlir::memref::LoadOp>(
                    body_loc, conv_output, mlir::ValueRange{iv_n, iv_oc, iv_oh, iv_ow}).getResult();
            }
            auto for_ic = b.create<mlir::scf::ForOp>(
                body_loc, c0, C_in, c1, acc0,
                [&](mlir::OpBuilder& b1, mlir::Location loc1, mlir::Value iv_ic, mlir::ValueRange iter_args) {
                    auto acc_ic = iter_args[0];
                    auto for_kh = b1.create<mlir::scf::ForOp>(
                        loc1, c0, kH, c1, acc_ic,
                        [&](mlir::OpBuilder& b2, mlir::Location loc2, mlir::Value iv_kh, mlir::ValueRange iter_args2) {
                            auto acc_kh = iter_args2[0];
                            auto for_kw = b2.create<mlir::scf::ForOp>(
                                loc2, c0, kW, c1, acc_kh,
                                [&](mlir::OpBuilder& b3, mlir::Location loc3, mlir::Value iv_kw, mlir::ValueRange iter_args3) {
                                    auto acc_kw = iter_args3[0];
                                    auto oh_mul = b3.create<mlir::arith::MulIOp>(loc3, iv_oh, strideH);
                                    auto ow_mul = b3.create<mlir::arith::MulIOp>(loc3, iv_ow, strideW);
                                    auto kh_mul = b3.create<mlir::arith::MulIOp>(loc3, iv_kh, dilH);
                                    auto kw_mul = b3.create<mlir::arith::MulIOp>(loc3, iv_kw, dilW);
                                    mlir::Value ih = b3.create<mlir::arith::AddIOp>(loc3, oh_mul, kh_mul).getResult();
                                    mlir::Value iw = b3.create<mlir::arith::AddIOp>(loc3, ow_mul, kw_mul).getResult();
                                    if (needs_bounds) {
                                        ih = b3.create<mlir::arith::SubIOp>(loc3, ih, padH).getResult();
                                        iw = b3.create<mlir::arith::SubIOp>(loc3, iw, padW).getResult();
                                    }

                                    auto w_val = b3.create<mlir::memref::LoadOp>(
                                        loc3, filter, mlir::ValueRange{iv_oc, iv_ic, iv_kh, iv_kw}).getResult();
                                    mlir::Value acc_next = acc_kw;
                                    if (needs_bounds) {
                                        auto ge_h = b3.create<mlir::arith::CmpIOp>(
                                            loc3, mlir::arith::CmpIPredicate::sge, ih, c0);
                                        auto lt_h = b3.create<mlir::arith::CmpIOp>(
                                            loc3, mlir::arith::CmpIPredicate::slt, ih, H_in);
                                        auto ge_w = b3.create<mlir::arith::CmpIOp>(
                                            loc3, mlir::arith::CmpIPredicate::sge, iw, c0);
                                        auto lt_w = b3.create<mlir::arith::CmpIOp>(
                                            loc3, mlir::arith::CmpIPredicate::slt, iw, W_in);
                                        auto in_h = b3.create<mlir::arith::AndIOp>(loc3, ge_h, lt_h);
                                        auto in_w = b3.create<mlir::arith::AndIOp>(loc3, ge_w, lt_w);
                                        auto in_bounds = b3.create<mlir::arith::AndIOp>(loc3, in_h, in_w);
                                        auto ifop = b3.create<mlir::scf::IfOp>(
                                            loc3, acc_kw.getType(), in_bounds, /*withElse=*/true);
                                        {
                                            mlir::OpBuilder::InsertionGuard guard(b3);
                                            b3.setInsertionPointToStart(&ifop.getThenRegion().front());
                                            auto in_val = b3.create<mlir::memref::LoadOp>(
                                                loc3, conv_input, mlir::ValueRange{iv_n, iv_ic, ih, iw}).getResult();
                                            auto mul = b3.create<mlir::arith::MulFOp>(loc3, in_val, w_val);
                                            auto add = b3.create<mlir::arith::AddFOp>(loc3, acc_kw, mul).getResult();
                                            b3.create<mlir::scf::YieldOp>(loc3, mlir::ValueRange{add});
                                        }
                                        {
                                            mlir::OpBuilder::InsertionGuard guard(b3);
                                            b3.setInsertionPointToStart(&ifop.getElseRegion().front());
                                            b3.create<mlir::scf::YieldOp>(loc3, mlir::ValueRange{acc_kw});
                                        }
                                        acc_next = ifop.getResult(0);
                                    } else {
                                        auto in_val = b3.create<mlir::memref::LoadOp>(
                                            loc3, conv_input, mlir::ValueRange{iv_n, iv_ic, ih, iw}).getResult();
                                        auto mul = b3.create<mlir::arith::MulFOp>(loc3, in_val, w_val);
                                        acc_next = b3.create<mlir::arith::AddFOp>(loc3, acc_kw, mul).getResult();
                                    }
                                    b3.create<mlir::scf::YieldOp>(loc3, mlir::ValueRange{acc_next});
                                });
                            b2.create<mlir::scf::YieldOp>(loc2, mlir::ValueRange{for_kw.getResult(0)});
                        });
                    b1.create<mlir::scf::YieldOp>(loc1, mlir::ValueRange{for_kh.getResult(0)});
                });
            auto acc_final = for_ic.getResult(0);
            b.create<mlir::memref::StoreOp>(body_loc, acc_final, conv_output,
                                            mlir::ValueRange{iv_n, iv_oc, iv_oh, iv_ow});
            b.create<mlir::scf::YieldOp>(body_loc);
        });

    if (op->getNumResults() > 0) {
        rewriter.replaceOp(op, conv_output);
    } else {
        rewriter.eraseOp(op);
    }
    if (fill_op) {
        rewriter.eraseOp(fill_op);
    }
    if (pad_fill_parallel && !using_padded_input) {
        rewriter.eraseOp(pad_fill_parallel);
    }
    return true;
}

}  // namespace

void run_conv2d_parallel_lowering(mlir::ModuleOp module) {
    if (!module) {
        return;
    }
    mlir::IRRewriter rewriter(module.getContext());
    llvm::SmallVector<mlir::linalg::Conv2DNchwFchwOp, 8> convs;
    module.walk([&](mlir::linalg::Conv2DNchwFchwOp op) {
        convs.push_back(op);
    });
    size_t rewritten = 0;
    for (auto op : convs) {
        if (!op || !op->getParentOp()) {
            continue;
        }
        if (lower_conv2d_op(op, rewriter)) {
            ++rewritten;
        }
    }
    if (gfx_log_debug_enabled()) {
        GFX_LOG_DEBUG("MLIR", "Conv2D parallel lowering: convs=" << convs.size()
                                                                 << " rewritten=" << rewritten);
    }
    if (!convs.empty() && rewritten == 0) {
        throw std::runtime_error("Conv2D parallel lowering failed");
    }
}

}  // namespace gfx_plugin
}  // namespace ov
