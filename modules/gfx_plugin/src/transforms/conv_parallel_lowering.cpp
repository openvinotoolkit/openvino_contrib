// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transforms/conv_parallel_lowering.hpp"

#include <optional>
#include <stdexcept>
#include <string>

#include "runtime/gfx_activation.hpp"
#include "runtime/gfx_logger.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

namespace ov {
namespace gfx_plugin {

namespace {

std::optional<ActivationKind> parse_activation_kind(mlir::Operation* op) {
    if (!op) {
        return std::nullopt;
    }
    auto attr = op->getAttrOfType<mlir::StringAttr>("gfx.activation_kind");
    if (!attr) {
        return std::nullopt;
    }
    const auto name = attr.getValue();
    if (name == "Relu") return ActivationKind::Relu;
    if (name == "Sigmoid") return ActivationKind::Sigmoid;
    if (name == "Tanh") return ActivationKind::Tanh;
    if (name == "Elu") return ActivationKind::Elu;
    if (name == "Prelu") return ActivationKind::Prelu;
    if (name == "Gelu") return ActivationKind::Gelu;
    if (name == "Swish") return ActivationKind::Swish;
    if (name == "HSwish") return ActivationKind::HSwish;
    if (name == "HSigmoid") return ActivationKind::HSigmoid;
    if (name == "Abs") return ActivationKind::Abs;
    if (name == "Sign") return ActivationKind::Sign;
    return std::nullopt;
}

mlir::Value apply_activation(mlir::OpBuilder& b,
                             mlir::Location loc,
                             mlir::Value x,
                             ActivationKind kind,
                             float alpha,
                             mlir::Type elem_ty) {
    auto make_float_attr = [&](double v) { return mlir::FloatAttr::get(elem_ty, v); };
    auto zero = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(0.0));
    switch (kind) {
        case ActivationKind::Relu:
            return b.create<mlir::arith::MaximumFOp>(loc, x, zero);
        case ActivationKind::Sigmoid: {
            auto neg = b.create<mlir::arith::NegFOp>(loc, x);
            auto exp = b.create<mlir::math::ExpOp>(loc, neg);
            auto one = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(1.0));
            auto denom = b.create<mlir::arith::AddFOp>(loc, one, exp);
            return b.create<mlir::arith::DivFOp>(loc, one, denom);
        }
        case ActivationKind::Tanh:
            return b.create<mlir::math::TanhOp>(loc, x);
        case ActivationKind::Elu: {
            auto alpha_c = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(alpha));
            auto exp = b.create<mlir::math::ExpOp>(loc, x);
            auto one = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(1.0));
            auto expm1 = b.create<mlir::arith::SubFOp>(loc, exp, one);
            auto neg_branch = b.create<mlir::arith::MulFOp>(loc, alpha_c, expm1);
            auto cond = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, x, zero);
            return b.create<mlir::arith::SelectOp>(loc, cond, x, neg_branch);
        }
        case ActivationKind::Prelu: {
            auto alpha_c = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(alpha));
            auto neg_branch = b.create<mlir::arith::MulFOp>(loc, alpha_c, x);
            auto cond = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, x, zero);
            return b.create<mlir::arith::SelectOp>(loc, cond, x, neg_branch);
        }
        case ActivationKind::Gelu: {
            auto half = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(0.5));
            auto one = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(1.0));
            auto c0 = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(0.79788456));
            auto c1 = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(0.044715));
            auto x2 = b.create<mlir::arith::MulFOp>(loc, x, x);
            auto x3 = b.create<mlir::arith::MulFOp>(loc, x2, x);
            auto inner = b.create<mlir::arith::AddFOp>(loc, x, b.create<mlir::arith::MulFOp>(loc, c1, x3));
            auto tanh_arg = b.create<mlir::arith::MulFOp>(loc, c0, inner);
            auto tanh = b.create<mlir::math::TanhOp>(loc, tanh_arg);
            auto term = b.create<mlir::arith::AddFOp>(loc, one, tanh);
            auto mul = b.create<mlir::arith::MulFOp>(loc, half, b.create<mlir::arith::MulFOp>(loc, x, term));
            return mul;
        }
        case ActivationKind::Swish: {
            auto neg = b.create<mlir::arith::NegFOp>(loc, x);
            auto exp = b.create<mlir::math::ExpOp>(loc, neg);
            auto one = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(1.0));
            auto denom = b.create<mlir::arith::AddFOp>(loc, one, exp);
            auto sigmoid = b.create<mlir::arith::DivFOp>(loc, one, denom);
            return b.create<mlir::arith::MulFOp>(loc, x, sigmoid);
        }
        case ActivationKind::HSwish:
        case ActivationKind::HSigmoid: {
            auto three = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(3.0));
            auto six = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(6.0));
            auto inv6 = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(1.0 / 6.0));
            auto x_plus = b.create<mlir::arith::AddFOp>(loc, x, three);
            auto max0 = b.create<mlir::arith::MaximumFOp>(loc, x_plus, zero);
            auto min6 = b.create<mlir::arith::MinimumFOp>(loc, max0, six);
            auto hsig = b.create<mlir::arith::MulFOp>(loc, min6, inv6);
            if (kind == ActivationKind::HSwish) {
                return b.create<mlir::arith::MulFOp>(loc, x, hsig);
            }
            return hsig;
        }
        case ActivationKind::Abs: {
            auto cond = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, x, zero);
            auto neg = b.create<mlir::arith::NegFOp>(loc, x);
            return b.create<mlir::arith::SelectOp>(loc, cond, neg, x);
        }
        case ActivationKind::Sign: {
            auto one = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(1.0));
            auto neg_one = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(-1.0));
            auto gt = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, x, zero);
            auto lt = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, x, zero);
            auto pos = b.create<mlir::arith::SelectOp>(loc, gt, one, zero);
            return b.create<mlir::arith::SelectOp>(loc, lt, neg_one, pos);
        }
        default:
            break;
    }
    return x;
}

struct BnGlobals {
    mlir::Value scale;
    mlir::Value bias;
};

mlir::Value append_func_arg(mlir::func::FuncOp func, mlir::Type type, mlir::Location loc) {
    auto fn_type = func.getFunctionType();
    llvm::SmallVector<mlir::Type, 8> inputs(fn_type.getInputs().begin(), fn_type.getInputs().end());
    inputs.push_back(type);
    auto new_type = mlir::FunctionType::get(func.getContext(), inputs, fn_type.getResults());
    func.setType(new_type);
    return func.getBody().front().addArgument(type, loc);
}

std::optional<mlir::Value> prepare_bias_global(mlir::linalg::Conv2DNchwFchwOp op,
                                               mlir::IRRewriter& rewriter,
                                               mlir::Type elem_ty) {
    auto bias_attr = op->getAttrOfType<mlir::DenseFPElementsAttr>("gfx.bias");
    if (!bias_attr) {
        return std::nullopt;
    }
    auto bias_type = mlir::dyn_cast<mlir::RankedTensorType>(bias_attr.getType());
    if (!bias_type || bias_type.getRank() != 1) {
        return std::nullopt;
    }
    auto func = op->getParentOfType<mlir::func::FuncOp>();
    if (!func) {
        return std::nullopt;
    }
    auto loc = op.getLoc();
    auto memref_type = mlir::MemRefType::get(bias_type.getShape(), elem_ty);
    return append_func_arg(func, memref_type, loc);
}

std::optional<BnGlobals> prepare_bn_globals(mlir::linalg::Conv2DNchwFchwOp op,
                                            mlir::IRRewriter& rewriter,
                                            mlir::Type elem_ty) {
    auto scale_attr = op->getAttrOfType<mlir::DenseFPElementsAttr>("gfx.bn_scale");
    auto bias_attr = op->getAttrOfType<mlir::DenseFPElementsAttr>("gfx.bn_bias");
    if (!scale_attr || !bias_attr) {
        return std::nullopt;
    }
    auto scale_type = mlir::dyn_cast<mlir::RankedTensorType>(scale_attr.getType());
    auto bias_type = mlir::dyn_cast<mlir::RankedTensorType>(bias_attr.getType());
    if (!scale_type || !bias_type || scale_type.getShape() != bias_type.getShape()) {
        return std::nullopt;
    }
    auto func = op->getParentOfType<mlir::func::FuncOp>();
    if (!func) {
        return std::nullopt;
    }
    auto loc = op.getLoc();
    auto memref_type = mlir::MemRefType::get(scale_type.getShape(), elem_ty);
    auto scale_arg = append_func_arg(func, memref_type, loc);
    auto bias_arg = append_func_arg(func, memref_type, loc);
    return BnGlobals{scale_arg, bias_arg};
}

bool extract_hw(mlir::DenseIntElementsAttr attr, int64_t& h, int64_t& w) {
    if (!attr) {
        return false;
    }
    const auto count = static_cast<size_t>(attr.getNumElements());
    if (count < 2) {
        return false;
    }
    auto it = attr.getValues<int64_t>().begin();
    for (size_t i = 0; i + 2 < count; ++i) {
        ++it;
    }
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
    mlir::Operation* pad_fill_loop = nullptr;
    mlir::Operation* pad_copy_loop = nullptr;
    const bool has_pad_begin_attr = op->getAttr("gfx.pad_begin") != nullptr;
    if (auto pad_attr = op->getAttrOfType<mlir::DenseIntElementsAttr>("gfx.pad_begin")) {
        (void)extract_hw(pad_attr, pad_h, pad_w);
    }
    if (auto pad_attr = op->getAttrOfType<mlir::DenseIntElementsAttr>("gfx.pad_end")) {
        (void)extract_hw(pad_attr, pad_end_h, pad_end_w);
    }

    bool found_copy = false;
    auto find_outer_loop = [](mlir::Operation* op) -> mlir::Operation* {
        mlir::Operation* outer = nullptr;
        for (auto* cur = op; cur; cur = cur->getParentOp()) {
            if (mlir::isa<mlir::scf::ForOp, mlir::scf::ParallelOp>(cur)) {
                outer = cur;
            }
        }
        return outer;
    };
    if (auto func = op->getParentOfType<mlir::func::FuncOp>()) {
        func.walk([&](mlir::memref::StoreOp store) {
            if (strip_memref_casts(store.getMemRef()) != input_base) {
                return;
            }
            if (!found_copy) {
                if (auto load = store.getValue().getDefiningOp<mlir::memref::LoadOp>()) {
                    conv_input = strip_memref_casts(load.getMemRef());
                    auto indices = store.getIndices();
                    if (!has_pad_begin_attr && indices.size() >= 4) {
                        (void)extract_addi_offset(indices[2], pad_h);
                        (void)extract_addi_offset(indices[3], pad_w);
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

    const bool input_is_padded = (pad_fill_loop != nullptr || pad_copy_loop != nullptr);
    const bool using_padded_input = input_is_padded && (strip_memref_casts(conv_input) == input_base);
    bool prefer_parallel = true;
    if (auto module = op->getParentOfType<mlir::ModuleOp>()) {
        if (auto attr = module->getAttrOfType<mlir::BoolAttr>("gfx.prefer_parallel")) {
            prefer_parallel = attr.getValue();
        }
    }
    const bool has_explicit_padding = (pad_h != 0 || pad_w != 0 || pad_end_h != 0 || pad_end_w != 0 ||
                                       pad_fill_loop != nullptr || pad_copy_loop != nullptr);
    if (!prefer_parallel && !has_explicit_padding) {
        return false;
    }
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
    constexpr int64_t kThreadH = 4;
    constexpr int64_t kThreadW = 4;
    // Keep micro-tiling minimal to reduce SPIR-V kernel complexity for Vulkan.
    int64_t micro_h = 1;
    int64_t micro_w = 1;
    const int64_t tile_h = kThreadH * micro_h;
    const int64_t tile_w = kThreadW * micro_w;
    if (auto module = op->getParentOfType<mlir::ModuleOp>()) {
        auto* ctx = module.getContext();
        module->setAttr("gfx.dispatch_tile_h",
                        mlir::IntegerAttr::get(mlir::IndexType::get(ctx), tile_h));
        module->setAttr("gfx.dispatch_tile_w",
                        mlir::IntegerAttr::get(mlir::IndexType::get(ctx), tile_w));
        module->setAttr("gfx.dispatch_threads_h",
                        mlir::IntegerAttr::get(mlir::IndexType::get(ctx), kThreadH));
        module->setAttr("gfx.dispatch_threads_w",
                        mlir::IntegerAttr::get(mlir::IndexType::get(ctx), kThreadW));
    }
    auto tileH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, tile_h);
    auto tileW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, tile_w);
    auto threadH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, kThreadH);
    auto threadW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, kThreadW);
    auto microH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, micro_h);
    auto microW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, micro_w);
    auto tileH_minus1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, tile_h - 1);
    auto tileW_minus1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, tile_w - 1);
    auto strideH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, stride_h);
    auto strideW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, stride_w);
    auto dilH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, dil_h);
    auto dilW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, dil_w);

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
    auto H_out = get_dim(conv_output, 2);
    auto W_out = get_dim(conv_output, 3);
    auto C_in = get_dim(conv_input, 1);
    auto H_in = get_dim(conv_input, 2);
    auto W_in = get_dim(conv_input, 3);
    auto kH = get_dim(filter, 2);
    auto kW = get_dim(filter, 3);

    // Map parallel loops to output C/H/W tiles with thread-level micro-tiles,
    // so Vulkan dispatch grid maps blocks to [C_out, H_tiles, W_tiles].
    const auto activation = parse_activation_kind(op);
    float activation_alpha = 0.0f;
    if (auto alpha_attr = op->getAttrOfType<mlir::FloatAttr>("gfx.activation_alpha")) {
        activation_alpha = static_cast<float>(alpha_attr.getValueAsDouble());
    }
    auto bias_global = prepare_bias_global(op, rewriter, elem_ty);
    auto bn_globals = prepare_bn_globals(op, rewriter, elem_ty);

    auto h_tiles_num = rewriter.create<mlir::arith::AddIOp>(loc, H_out, tileH_minus1);
    auto w_tiles_num = rewriter.create<mlir::arith::AddIOp>(loc, W_out, tileW_minus1);
    auto H_tiles = rewriter.create<mlir::arith::DivSIOp>(loc, h_tiles_num, tileH);
    auto W_tiles = rewriter.create<mlir::arith::DivSIOp>(loc, w_tiles_num, tileW);

    auto par = rewriter.create<mlir::scf::ParallelOp>(
        loc,
        mlir::ValueRange{c0, c0, c0, c0, c0},
        mlir::ValueRange{C_out, H_tiles, W_tiles, threadH, threadW},
        mlir::ValueRange{c1, c1, c1, c1, c1});

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(par.getBody()->getTerminator());

    auto ivs = par.getInductionVars();
    auto iv_oc = ivs[0];
    auto iv_oh_base = rewriter.create<mlir::arith::MulIOp>(loc, ivs[1], tileH);
    auto iv_ow_base = rewriter.create<mlir::arith::MulIOp>(loc, ivs[2], tileW);
    auto iv_th = ivs[3];
    auto iv_tw = ivs[4];
    auto iv_oh_off = rewriter.create<mlir::arith::MulIOp>(loc, iv_th, microH);
    auto iv_ow_off = rewriter.create<mlir::arith::MulIOp>(loc, iv_tw, microW);
    llvm::SmallVector<mlir::Value, 4> oh_vals;
    llvm::SmallVector<mlir::Value, 4> oh_in_vals;
    oh_vals.reserve(static_cast<size_t>(micro_h));
    oh_in_vals.reserve(static_cast<size_t>(micro_h));
    for (int64_t mh = 0; mh < micro_h; ++mh) {
        mlir::Value off = iv_oh_off;
        if (mh != 0) {
            off = rewriter.create<mlir::arith::AddIOp>(
                loc, iv_oh_off, rewriter.create<mlir::arith::ConstantIndexOp>(loc, mh));
        }
        auto oh = rewriter.create<mlir::arith::AddIOp>(loc, iv_oh_base, off);
        oh_vals.push_back(oh);
        oh_in_vals.push_back(rewriter.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::slt, oh, H_out));
    }
    llvm::SmallVector<mlir::Value, 8> ow_vals;
    llvm::SmallVector<mlir::Value, 8> ow_in_vals;
    ow_vals.reserve(static_cast<size_t>(micro_w));
    ow_in_vals.reserve(static_cast<size_t>(micro_w));
    for (int64_t mw = 0; mw < micro_w; ++mw) {
        mlir::Value off = iv_ow_off;
        if (mw != 0) {
            off = rewriter.create<mlir::arith::AddIOp>(
                loc, iv_ow_off, rewriter.create<mlir::arith::ConstantIndexOp>(loc, mw));
        }
        auto ow = rewriter.create<mlir::arith::AddIOp>(loc, iv_ow_base, off);
        ow_vals.push_back(ow);
        ow_in_vals.push_back(rewriter.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::slt, ow, W_out));
    }
    const bool needs_bounds = has_explicit_padding && !using_padded_input;
    const int64_t effective_pad_h = needs_bounds ? pad_h : 0;
    const int64_t effective_pad_w = needs_bounds ? pad_w : 0;
    auto padH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, effective_pad_h);
    auto padW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, effective_pad_w);
    rewriter.create<mlir::scf::ForOp>(
        loc, c0, N, c1, mlir::ValueRange{},
        [&](mlir::OpBuilder& b, mlir::Location body_loc, mlir::Value iv_n, mlir::ValueRange) {
            const int64_t lane_count = micro_h * micro_w;
            llvm::SmallVector<mlir::Value, 8> lane_in;
            llvm::SmallVector<mlir::Value, 8> lane_oh;
            llvm::SmallVector<mlir::Value, 8> lane_ow;
            lane_in.reserve(static_cast<size_t>(lane_count));
            lane_oh.reserve(static_cast<size_t>(lane_count));
            lane_ow.reserve(static_cast<size_t>(lane_count));
            for (int64_t mh = 0; mh < micro_h; ++mh) {
                for (int64_t mw = 0; mw < micro_w; ++mw) {
                    auto in = b.create<mlir::arith::AndIOp>(body_loc, oh_in_vals[mh], ow_in_vals[mw]);
                    lane_in.push_back(in);
                    lane_oh.push_back(oh_vals[mh]);
                    lane_ow.push_back(ow_vals[mw]);
                }
            }
            mlir::Value tile_in = lane_in.front();
            for (size_t i = 1; i < lane_in.size(); ++i) {
                tile_in = b.create<mlir::arith::OrIOp>(body_loc, tile_in, lane_in[i]);
            }

            auto if_tile = b.create<mlir::scf::IfOp>(body_loc, tile_in, /*withElse=*/false);
            {
                mlir::OpBuilder::InsertionGuard guard(b);
                b.setInsertionPointToStart(&if_tile.getThenRegion().front());
                auto zero = b.create<mlir::arith::ConstantOp>(
                    body_loc, elem_ty, b.getFloatAttr(elem_ty, 0.0));
                llvm::SmallVector<mlir::Value, 8> acc_init(static_cast<size_t>(lane_count), zero);
                if (!zero_init) {
                    for (int64_t i = 0; i < lane_count; ++i) {
                        auto if_acc = b.create<mlir::scf::IfOp>(
                            body_loc, acc_init[i].getType(), lane_in[i], /*withElse=*/true);
                        {
                            mlir::OpBuilder::InsertionGuard guard(b);
                            b.setInsertionPointToStart(&if_acc.getThenRegion().front());
                            auto v = b.create<mlir::memref::LoadOp>(
                                body_loc, conv_output, mlir::ValueRange{iv_n, iv_oc, lane_oh[i], lane_ow[i]}).getResult();
                            b.create<mlir::scf::YieldOp>(body_loc, mlir::ValueRange{v});
                        }
                        {
                            mlir::OpBuilder::InsertionGuard guard(b);
                            b.setInsertionPointToStart(&if_acc.getElseRegion().front());
                            b.create<mlir::scf::YieldOp>(body_loc, mlir::ValueRange{zero});
                        }
                        acc_init[i] = if_acc.getResult(0);
                    }
                }

                auto for_ic = b.create<mlir::scf::ForOp>(
                    body_loc, c0, C_in, c1, acc_init,
                    [&](mlir::OpBuilder& b3, mlir::Location loc3, mlir::Value iv_ic, mlir::ValueRange iter_args) {
                        llvm::SmallVector<mlir::Value, 8> acc_ic(iter_args.begin(), iter_args.end());
                        auto for_kh = b3.create<mlir::scf::ForOp>(
                            loc3, c0, kH, c1, acc_ic,
                            [&](mlir::OpBuilder& b4, mlir::Location loc4, mlir::Value iv_kh, mlir::ValueRange iter_args2) {
                                llvm::SmallVector<mlir::Value, 8> acc_kh(iter_args2.begin(), iter_args2.end());
                                auto for_kw = b4.create<mlir::scf::ForOp>(
                                    loc4, c0, kW, c1, acc_kh,
                                    [&](mlir::OpBuilder& b5, mlir::Location loc5, mlir::Value iv_kw, mlir::ValueRange iter_args3) {
                                        llvm::SmallVector<mlir::Value, 8> acc_kw(iter_args3.begin(), iter_args3.end());
                                        auto kh_mul = b5.create<mlir::arith::MulIOp>(loc5, iv_kh, dilH);
                                        auto kw_mul = b5.create<mlir::arith::MulIOp>(loc5, iv_kw, dilW);
                                        auto w_val = b5.create<mlir::memref::LoadOp>(
                                            loc5, filter, mlir::ValueRange{iv_oc, iv_ic, iv_kh, iv_kw}).getResult();
                                        llvm::SmallVector<mlir::Value, 8> next_accs;
                                        next_accs.reserve(acc_kw.size());
                                        for (int64_t i = 0; i < lane_count; ++i) {
                                            auto oh_mul = b5.create<mlir::arith::MulIOp>(loc5, lane_oh[i], strideH);
                                            auto ow_mul = b5.create<mlir::arith::MulIOp>(loc5, lane_ow[i], strideW);
                                            mlir::Value ih_padded = b5.create<mlir::arith::AddIOp>(loc5, oh_mul, kh_mul).getResult();
                                            mlir::Value iw_padded = b5.create<mlir::arith::AddIOp>(loc5, ow_mul, kw_mul).getResult();
                                            auto if_lane = b5.create<mlir::scf::IfOp>(
                                                loc5, acc_kw[i].getType(), lane_in[i], /*withElse=*/true);
                                            {
                                                mlir::OpBuilder::InsertionGuard guard(b5);
                                                b5.setInsertionPointToStart(&if_lane.getThenRegion().front());
                                                mlir::Value acc_next = acc_kw[i];
                                                if (needs_bounds) {
                                                    auto ih = b5.create<mlir::arith::SubIOp>(loc5, ih_padded, padH).getResult();
                                                    auto iw = b5.create<mlir::arith::SubIOp>(loc5, iw_padded, padW).getResult();
                                                    auto ge_h = b5.create<mlir::arith::CmpIOp>(
                                                        loc5, mlir::arith::CmpIPredicate::sge, ih, c0);
                                                    auto lt_h = b5.create<mlir::arith::CmpIOp>(
                                                        loc5, mlir::arith::CmpIPredicate::slt, ih, H_in);
                                                    auto ge_w = b5.create<mlir::arith::CmpIOp>(
                                                        loc5, mlir::arith::CmpIPredicate::sge, iw, c0);
                                                    auto lt_w = b5.create<mlir::arith::CmpIOp>(
                                                        loc5, mlir::arith::CmpIPredicate::slt, iw, W_in);
                                                    auto in_h2 = b5.create<mlir::arith::AndIOp>(loc5, ge_h, lt_h);
                                                    auto in_w2 = b5.create<mlir::arith::AndIOp>(loc5, ge_w, lt_w);
                                                    auto in_bounds2 = b5.create<mlir::arith::AndIOp>(loc5, in_h2, in_w2);
                                                    auto ifop2 = b5.create<mlir::scf::IfOp>(
                                                        loc5, acc_kw[i].getType(), in_bounds2, /*withElse=*/true);
                                                    {
                                                        mlir::OpBuilder::InsertionGuard guard(b5);
                                                        b5.setInsertionPointToStart(&ifop2.getThenRegion().front());
                                                        auto in_val = b5.create<mlir::memref::LoadOp>(
                                                            loc5, conv_input, mlir::ValueRange{iv_n, iv_ic, ih, iw}).getResult();
                                                        auto mul = b5.create<mlir::arith::MulFOp>(loc5, in_val, w_val);
                                                        auto add = b5.create<mlir::arith::AddFOp>(loc5, acc_kw[i], mul).getResult();
                                                        b5.create<mlir::scf::YieldOp>(loc5, mlir::ValueRange{add});
                                                    }
                                                    {
                                                        mlir::OpBuilder::InsertionGuard guard(b5);
                                                        b5.setInsertionPointToStart(&ifop2.getElseRegion().front());
                                                        b5.create<mlir::scf::YieldOp>(loc5, mlir::ValueRange{acc_kw[i]});
                                                    }
                                                    acc_next = ifop2.getResult(0);
                                                } else {
                                                    auto in_val = b5.create<mlir::memref::LoadOp>(
                                                        loc5, conv_input, mlir::ValueRange{iv_n, iv_ic, ih_padded, iw_padded}).getResult();
                                                    auto mul = b5.create<mlir::arith::MulFOp>(loc5, in_val, w_val);
                                                    acc_next = b5.create<mlir::arith::AddFOp>(loc5, acc_kw[i], mul).getResult();
                                                }
                                                b5.create<mlir::scf::YieldOp>(loc5, mlir::ValueRange{acc_next});
                                            }
                                            {
                                                mlir::OpBuilder::InsertionGuard guard(b5);
                                                b5.setInsertionPointToStart(&if_lane.getElseRegion().front());
                                                b5.create<mlir::scf::YieldOp>(loc5, mlir::ValueRange{acc_kw[i]});
                                            }
                                            next_accs.push_back(if_lane.getResult(0));
                                        }
                                        b5.create<mlir::scf::YieldOp>(loc5, next_accs);
                                    });
                                b4.create<mlir::scf::YieldOp>(loc4, for_kw.getResults());
                            });
                        b3.create<mlir::scf::YieldOp>(loc3, for_kh.getResults());
                    });

                llvm::SmallVector<mlir::Value, 8> acc_final(for_ic.getResults().begin(), for_ic.getResults().end());
                auto apply_post = [&](mlir::Value acc) -> mlir::Value {
                    if (bn_globals.has_value()) {
                        auto scale = b.create<mlir::memref::LoadOp>(
                            body_loc, bn_globals->scale, mlir::ValueRange{iv_oc}).getResult();
                        auto bias = b.create<mlir::memref::LoadOp>(
                            body_loc, bn_globals->bias, mlir::ValueRange{iv_oc}).getResult();
                        auto mul = b.create<mlir::arith::MulFOp>(body_loc, acc, scale).getResult();
                        acc = b.create<mlir::arith::AddFOp>(body_loc, mul, bias).getResult();
                    }
                    if (bias_global.has_value()) {
                        auto bias = b.create<mlir::memref::LoadOp>(
                            body_loc, *bias_global, mlir::ValueRange{iv_oc}).getResult();
                        acc = b.create<mlir::arith::AddFOp>(body_loc, acc, bias).getResult();
                    }
                    if (activation.has_value()) {
                        acc = apply_activation(b, body_loc, acc,
                                               *activation, activation_alpha, elem_ty);
                    }
                    return acc;
                };

                for (int64_t i = 0; i < lane_count; ++i) {
                    acc_final[i] = apply_post(acc_final[i]);
                }

                for (int64_t i = 0; i < lane_count; ++i) {
                    auto if_store = b.create<mlir::scf::IfOp>(
                        body_loc, lane_in[i], /*withElse=*/false);
                    {
                        mlir::OpBuilder::InsertionGuard guard(b);
                        b.setInsertionPointToStart(&if_store.getThenRegion().front());
                        b.create<mlir::memref::StoreOp>(
                            body_loc, acc_final[i], conv_output, mlir::ValueRange{iv_n, iv_oc, lane_oh[i], lane_ow[i]});
                    }
                }
            }
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
