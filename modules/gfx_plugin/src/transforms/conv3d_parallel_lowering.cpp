// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transforms/conv3d_parallel_lowering.hpp"

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

bool extract_dhw(mlir::DenseIntElementsAttr attr, int64_t& d, int64_t& h, int64_t& w) {
    if (!attr) {
        return false;
    }
    const auto count = static_cast<size_t>(attr.getNumElements());
    if (count < 3) {
        return false;
    }
    auto it = attr.getValues<int64_t>().begin();
    for (size_t i = 0; i + 3 < count; ++i) {
        ++it;
    }
    d = *it++;
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

mlir::Operation* find_outer_loop(mlir::Operation* op) {
    mlir::Operation* outer = nullptr;
    for (auto* cur = op; cur; cur = cur->getParentOp()) {
        if (mlir::isa<mlir::scf::ForOp, mlir::scf::ParallelOp>(cur)) {
            outer = cur;
        }
    }
    return outer;
}

bool lower_conv3d_op(mlir::linalg::Conv3DNcdhwFcdhwOp op, mlir::IRRewriter& rewriter) {
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
        return false;
    }
    if (in_type.getRank() != 5 || w_type.getRank() != 5 || out_type.getRank() != 5) {
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
    int64_t pad_d = 0;
    int64_t pad_h = 0;
    int64_t pad_w = 0;
    int64_t pad_end_d = 0;
    int64_t pad_end_h = 0;
    int64_t pad_end_w = 0;
    mlir::Operation* pad_fill_loop = nullptr;
    mlir::Operation* pad_copy_loop = nullptr;
    const bool has_pad_begin_attr = op->getAttr("gfx.pad_begin") != nullptr;
    if (auto pad_attr = op->getAttrOfType<mlir::DenseIntElementsAttr>("gfx.pad_begin")) {
        (void)extract_dhw(pad_attr, pad_d, pad_h, pad_w);
    }
    if (auto pad_attr = op->getAttrOfType<mlir::DenseIntElementsAttr>("gfx.pad_end")) {
        (void)extract_dhw(pad_attr, pad_end_d, pad_end_h, pad_end_w);
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
                    if (!has_pad_begin_attr && indices.size() >= 5) {
                        (void)extract_addi_offset(indices[2], pad_d);
                        (void)extract_addi_offset(indices[3], pad_h);
                        (void)extract_addi_offset(indices[4], pad_w);
                    }
                    pad_copy_loop = find_outer_loop(store);
                    found_copy = true;
                    return;
                }
            }
            if (!pad_fill_loop) {
                if (auto cst = store.getValue().getDefiningOp<mlir::arith::ConstantOp>()) {
                    if (auto fattr = mlir::dyn_cast<mlir::FloatAttr>(cst.getValue())) {
                        if (fattr.getValueAsDouble() == 0.0) {
                            pad_fill_loop = find_outer_loop(store);
                        }
                    }
                }
            }
        });
    }

    if (auto func = op->getParentOfType<mlir::func::FuncOp>()) {
        auto padded_type = mlir::dyn_cast<mlir::MemRefType>(input_base.getType());
        if (padded_type && padded_type.getRank() == 5) {
            auto padded_shape = padded_type.getShape();
            if (padded_shape[2] != mlir::ShapedType::kDynamic &&
                padded_shape[3] != mlir::ShapedType::kDynamic &&
                padded_shape[4] != mlir::ShapedType::kDynamic) {
                for (auto arg : func.getArguments()) {
                    auto arg_type = mlir::dyn_cast<mlir::MemRefType>(arg.getType());
                    if (!arg_type || arg_type.getRank() != 5) {
                        continue;
                    }
                    if (arg_type.getElementType() != padded_type.getElementType()) {
                        continue;
                    }
                    auto arg_shape = arg_type.getShape();
                    if (arg_shape[0] != padded_shape[0] || arg_shape[1] != padded_shape[1]) {
                        continue;
                    }
                    if (arg_shape[2] == mlir::ShapedType::kDynamic ||
                        arg_shape[3] == mlir::ShapedType::kDynamic ||
                        arg_shape[4] == mlir::ShapedType::kDynamic) {
                        continue;
                    }
                    const int64_t cand_end_d = padded_shape[2] - pad_d - arg_shape[2];
                    const int64_t cand_end_h = padded_shape[3] - pad_h - arg_shape[3];
                    const int64_t cand_end_w = padded_shape[4] - pad_w - arg_shape[4];
                    if (cand_end_d < 0 || cand_end_h < 0 || cand_end_w < 0) {
                        continue;
                    }
                    pad_end_d = cand_end_d;
                    pad_end_h = cand_end_h;
                    pad_end_w = cand_end_w;
                    conv_input = strip_memref_casts(arg);
                    break;
                }
            }
        }
    }

    const bool input_is_padded = (pad_fill_loop != nullptr || pad_copy_loop != nullptr);
    const bool using_padded_input = input_is_padded && (strip_memref_casts(conv_input) == input_base);
    const bool has_explicit_padding = (pad_d != 0 || pad_h != 0 || pad_w != 0 ||
                                       pad_end_d != 0 || pad_end_h != 0 || pad_end_w != 0 ||
                                       pad_fill_loop != nullptr || pad_copy_loop != nullptr);
    if (!has_explicit_padding) {
        return false;
    }

    if (gfx_log_debug_enabled()) {
        GFX_LOG_DEBUG("MLIR", "Conv3D pad detect: conv_input="
                                  << (using_padded_input ? "padded" : "orig")
                                  << " pad_d=" << pad_d << " pad_h=" << pad_h << " pad_w=" << pad_w
                                  << " pad_end_d=" << pad_end_d
                                  << " pad_end_h=" << pad_end_h
                                  << " pad_end_w=" << pad_end_w);
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

    int64_t stride_d = 0, stride_h = 0, stride_w = 0;
    int64_t dil_d = 0, dil_h = 0, dil_w = 0;
    if (!extract_dhw(op.getStrides(), stride_d, stride_h, stride_w) ||
        !extract_dhw(op.getDilations(), dil_d, dil_h, dil_w)) {
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

    auto get_dim = [&](mlir::Value value, int64_t dim) -> mlir::Value {
        auto mem_ty = mlir::dyn_cast<mlir::MemRefType>(value.getType());
        if (mem_ty && dim < mem_ty.getRank()) {
            const int64_t sz = mem_ty.getDimSize(dim);
            if (sz != mlir::ShapedType::kDynamic) {
                return rewriter.create<mlir::arith::ConstantIndexOp>(loc, sz);
            }
        }
        return rewriter.create<mlir::memref::DimOp>(loc, value, dim);
    };

    auto N = get_dim(conv_output, 0);
    auto C_out = get_dim(conv_output, 1);
    auto D_out = get_dim(conv_output, 2);
    auto H_out = get_dim(conv_output, 3);
    auto W_out = get_dim(conv_output, 4);
    auto C_in = get_dim(conv_input, 1);
    auto D_in = get_dim(conv_input, 2);
    auto H_in = get_dim(conv_input, 3);
    auto W_in = get_dim(conv_input, 4);
    auto kD = get_dim(filter, 2);
    auto kH = get_dim(filter, 3);
    auto kW = get_dim(filter, 4);

    const bool needs_bounds = has_explicit_padding && !using_padded_input;
    const int64_t effective_pad_d = needs_bounds ? pad_d : 0;
    const int64_t effective_pad_h = needs_bounds ? pad_h : 0;
    const int64_t effective_pad_w = needs_bounds ? pad_w : 0;
    auto padD = rewriter.create<mlir::arith::ConstantIndexOp>(loc, effective_pad_d);
    auto padH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, effective_pad_h);
    auto padW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, effective_pad_w);
    auto strideD = rewriter.create<mlir::arith::ConstantIndexOp>(loc, stride_d);
    auto strideH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, stride_h);
    auto strideW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, stride_w);
    auto dilD = rewriter.create<mlir::arith::ConstantIndexOp>(loc, dil_d);
    auto dilH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, dil_h);
    auto dilW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, dil_w);

    rewriter.create<mlir::scf::ForOp>(
        loc, c0, N, c1, mlir::ValueRange{},
        [&](mlir::OpBuilder& b, mlir::Location loc1, mlir::Value iv_n, mlir::ValueRange) {
            b.create<mlir::scf::ForOp>(
                loc1, c0, C_out, c1, mlir::ValueRange{},
                [&](mlir::OpBuilder& b2, mlir::Location loc2, mlir::Value iv_oc, mlir::ValueRange) {
                    b2.create<mlir::scf::ForOp>(
                        loc2, c0, D_out, c1, mlir::ValueRange{},
                        [&](mlir::OpBuilder& b3, mlir::Location loc3, mlir::Value iv_od, mlir::ValueRange) {
                            b3.create<mlir::scf::ForOp>(
                                loc3, c0, H_out, c1, mlir::ValueRange{},
                                [&](mlir::OpBuilder& b4, mlir::Location loc4, mlir::Value iv_oh, mlir::ValueRange) {
                                    b4.create<mlir::scf::ForOp>(
                                        loc4, c0, W_out, c1, mlir::ValueRange{},
                                        [&](mlir::OpBuilder& b5, mlir::Location loc5, mlir::Value iv_ow, mlir::ValueRange) {
                                            mlir::Value acc_init;
                                            if (zero_init) {
                                                acc_init = zero_init;
                                            } else {
                                                acc_init = b5.create<mlir::memref::LoadOp>(
                                                    loc5, conv_output,
                                                    mlir::ValueRange{iv_n, iv_oc, iv_od, iv_oh, iv_ow}).getResult();
                                            }
                                            auto for_ic = b5.create<mlir::scf::ForOp>(
                                                loc5, c0, C_in, c1, mlir::ValueRange{acc_init},
                                                [&](mlir::OpBuilder& b6, mlir::Location loc6, mlir::Value iv_ic, mlir::ValueRange acc_ic_range) {
                                                    auto acc_ic = acc_ic_range.front();
                                                    auto for_kd = b6.create<mlir::scf::ForOp>(
                                                        loc6, c0, kD, c1, mlir::ValueRange{acc_ic},
                                                        [&](mlir::OpBuilder& b7, mlir::Location loc7, mlir::Value iv_kd, mlir::ValueRange acc_kd_range) {
                                                            auto acc_kd = acc_kd_range.front();
                                                            auto for_kh = b7.create<mlir::scf::ForOp>(
                                                                loc7, c0, kH, c1, mlir::ValueRange{acc_kd},
                                                                [&](mlir::OpBuilder& b8, mlir::Location loc8, mlir::Value iv_kh, mlir::ValueRange acc_kh_range) {
                                                                    auto acc_kh = acc_kh_range.front();
                                                                    auto for_kw = b8.create<mlir::scf::ForOp>(
                                                                        loc8, c0, kW, c1, mlir::ValueRange{acc_kh},
                                                                        [&](mlir::OpBuilder& b9, mlir::Location loc9, mlir::Value iv_kw, mlir::ValueRange acc_kw_range) {
                                                                            auto acc_kw = acc_kw_range.front();
                                                                            auto od_mul = b9.create<mlir::arith::MulIOp>(loc9, iv_od, strideD);
                                                                            auto kd_mul = b9.create<mlir::arith::MulIOp>(loc9, iv_kd, dilD);
                                                                            auto oh_mul = b9.create<mlir::arith::MulIOp>(loc9, iv_oh, strideH);
                                                                            auto kh_mul = b9.create<mlir::arith::MulIOp>(loc9, iv_kh, dilH);
                                                                            auto ow_mul = b9.create<mlir::arith::MulIOp>(loc9, iv_ow, strideW);
                                                                            auto kw_mul = b9.create<mlir::arith::MulIOp>(loc9, iv_kw, dilW);
                                                                            mlir::Value id_padded = b9.create<mlir::arith::AddIOp>(loc9, od_mul, kd_mul).getResult();
                                                                            mlir::Value ih_padded = b9.create<mlir::arith::AddIOp>(loc9, oh_mul, kh_mul).getResult();
                                                                            mlir::Value iw_padded = b9.create<mlir::arith::AddIOp>(loc9, ow_mul, kw_mul).getResult();
                                                                            mlir::Value acc_next = acc_kw;
                                                                            if (needs_bounds) {
                                                                                auto id = b9.create<mlir::arith::SubIOp>(loc9, id_padded, padD).getResult();
                                                                                auto ih = b9.create<mlir::arith::SubIOp>(loc9, ih_padded, padH).getResult();
                                                                                auto iw = b9.create<mlir::arith::SubIOp>(loc9, iw_padded, padW).getResult();
                                                                                auto ge_d = b9.create<mlir::arith::CmpIOp>(
                                                                                    loc9, mlir::arith::CmpIPredicate::sge, id, c0);
                                                                                auto lt_d = b9.create<mlir::arith::CmpIOp>(
                                                                                    loc9, mlir::arith::CmpIPredicate::slt, id, D_in);
                                                                                auto ge_h = b9.create<mlir::arith::CmpIOp>(
                                                                                    loc9, mlir::arith::CmpIPredicate::sge, ih, c0);
                                                                                auto lt_h = b9.create<mlir::arith::CmpIOp>(
                                                                                    loc9, mlir::arith::CmpIPredicate::slt, ih, H_in);
                                                                                auto ge_w = b9.create<mlir::arith::CmpIOp>(
                                                                                    loc9, mlir::arith::CmpIPredicate::sge, iw, c0);
                                                                                auto lt_w = b9.create<mlir::arith::CmpIOp>(
                                                                                    loc9, mlir::arith::CmpIPredicate::slt, iw, W_in);
                                                                                auto in_d = b9.create<mlir::arith::AndIOp>(loc9, ge_d, lt_d);
                                                                                auto in_h = b9.create<mlir::arith::AndIOp>(loc9, ge_h, lt_h);
                                                                                auto in_w = b9.create<mlir::arith::AndIOp>(loc9, ge_w, lt_w);
                                                                                auto in_dh = b9.create<mlir::arith::AndIOp>(loc9, in_d, in_h);
                                                                                auto in_bounds = b9.create<mlir::arith::AndIOp>(loc9, in_dh, in_w);
                                                                                auto ifop = b9.create<mlir::scf::IfOp>(
                                                                                    loc9, acc_kw.getType(), in_bounds, /*withElse=*/true);
                                                                                {
                                                                                    mlir::OpBuilder::InsertionGuard guard(b9);
                                                                                    b9.setInsertionPointToStart(&ifop.getThenRegion().front());
                                                                                    auto in_val = b9.create<mlir::memref::LoadOp>(
                                                                                        loc9, conv_input, mlir::ValueRange{iv_n, iv_ic, id, ih, iw}).getResult();
                                                                                    auto w_val = b9.create<mlir::memref::LoadOp>(
                                                                                        loc9, filter, mlir::ValueRange{iv_oc, iv_ic, iv_kd, iv_kh, iv_kw}).getResult();
                                                                                    auto mul = b9.create<mlir::arith::MulFOp>(loc9, in_val, w_val).getResult();
                                                                                    auto add = b9.create<mlir::arith::AddFOp>(loc9, acc_kw, mul).getResult();
                                                                                    b9.create<mlir::scf::YieldOp>(loc9, mlir::ValueRange{add});
                                                                                }
                                                                                {
                                                                                    mlir::OpBuilder::InsertionGuard guard(b9);
                                                                                    b9.setInsertionPointToStart(&ifop.getElseRegion().front());
                                                                                    b9.create<mlir::scf::YieldOp>(loc9, mlir::ValueRange{acc_kw});
                                                                                }
                                                                                acc_next = ifop.getResult(0);
                                                                            } else {
                                                                                auto in_val = b9.create<mlir::memref::LoadOp>(
                                                                                    loc9, conv_input, mlir::ValueRange{iv_n, iv_ic, id_padded, ih_padded, iw_padded}).getResult();
                                                                                auto w_val = b9.create<mlir::memref::LoadOp>(
                                                                                    loc9, filter, mlir::ValueRange{iv_oc, iv_ic, iv_kd, iv_kh, iv_kw}).getResult();
                                                                                auto mul = b9.create<mlir::arith::MulFOp>(loc9, in_val, w_val).getResult();
                                                                                acc_next = b9.create<mlir::arith::AddFOp>(loc9, acc_kw, mul).getResult();
                                                                            }
                                                                            b9.create<mlir::scf::YieldOp>(loc9, mlir::ValueRange{acc_next});
                                                                        });
                                                                    b8.create<mlir::scf::YieldOp>(loc8, for_kw.getResults());
                                                                });
                                                            b7.create<mlir::scf::YieldOp>(loc7, for_kh.getResults());
                                                        });
                                                    b6.create<mlir::scf::YieldOp>(loc6, for_kd.getResults());
                                                });

                                            auto acc_final = for_ic.getResult(0);
                                            b5.create<mlir::memref::StoreOp>(
                                                loc5, acc_final, conv_output,
                                                mlir::ValueRange{iv_n, iv_oc, iv_od, iv_oh, iv_ow});
                                            b5.create<mlir::scf::YieldOp>(loc5);
                                        });
                                    b4.create<mlir::scf::YieldOp>(loc4);
                                });
                            b3.create<mlir::scf::YieldOp>(loc3);
                        });
                    b2.create<mlir::scf::YieldOp>(loc2);
                });
            b.create<mlir::scf::YieldOp>(loc1);
        });

    if (op->getNumResults() > 0) {
        rewriter.replaceOp(op, conv_output);
    } else {
        rewriter.eraseOp(op);
    }
    if (fill_op) {
        rewriter.eraseOp(fill_op);
    }
    if (!using_padded_input) {
        if (pad_copy_loop && pad_copy_loop->getParentOp()) {
            rewriter.eraseOp(pad_copy_loop);
        }
        if (pad_fill_loop && pad_fill_loop->getParentOp() && pad_fill_loop != pad_copy_loop) {
            rewriter.eraseOp(pad_fill_loop);
        }
        if (auto alloc = input_base.getDefiningOp<mlir::memref::AllocOp>()) {
            if (alloc->use_empty()) {
                rewriter.eraseOp(alloc);
            }
        }
        if (auto alloca = input_base.getDefiningOp<mlir::memref::AllocaOp>()) {
            if (alloca->use_empty()) {
                rewriter.eraseOp(alloca);
            }
        }
    }
    return true;
}

}  // namespace

void run_conv3d_parallel_lowering(mlir::ModuleOp module) {
    if (!module) {
        return;
    }
    mlir::IRRewriter rewriter(module.getContext());
    llvm::SmallVector<mlir::linalg::Conv3DNcdhwFcdhwOp, 8> convs;
    module.walk([&](mlir::linalg::Conv3DNcdhwFcdhwOp op) {
        convs.push_back(op);
    });
    size_t rewritten = 0;
    for (auto op : convs) {
        if (!op || !op->getParentOp()) {
            continue;
        }
        if (lower_conv3d_op(op, rewriter)) {
            ++rewritten;
        }
    }
    if (gfx_log_debug_enabled()) {
        GFX_LOG_DEBUG("MLIR", "Conv3D lowering: convs=" << convs.size()
                                                        << " rewritten=" << rewritten);
    }
}

}  // namespace gfx_plugin
}  // namespace ov
