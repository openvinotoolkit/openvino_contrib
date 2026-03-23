// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "openvino/op/convolution.hpp"
#include "openvino/op/add.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/strides.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Casting.h"

namespace ov {
namespace gfx_plugin {

namespace {
mlir::SmallVector<int64_t> to_tensor_shape(const ov::PartialShape& ps) {
    mlir::SmallVector<int64_t> dims;
    dims.reserve(ps.rank().get_length());
    for (const auto& d : ps) {
        dims.push_back(d.is_dynamic() ? mlir::ShapedType::kDynamic
                                      : static_cast<int64_t>(d.get_length()));
    }
    return dims;
}

mlir::DenseIntElementsAttr make_i64_attr(mlir::OpBuilder& b, const ov::Strides& vals) {
    auto i64 = b.getI64Type();
    auto type = mlir::RankedTensorType::get({static_cast<int64_t>(vals.size())}, i64);
    llvm::SmallVector<int64_t, 4> data;
    data.reserve(vals.size());
    for (auto v : vals) {
        data.push_back(static_cast<int64_t>(v));
    }
    return mlir::DenseIntElementsAttr::get(type, data);
}

mlir::DenseIntElementsAttr make_i64_attr(mlir::OpBuilder& b, const ov::CoordinateDiff& vals) {
    auto i64 = b.getI64Type();
    auto type = mlir::RankedTensorType::get({static_cast<int64_t>(vals.size())}, i64);
    llvm::SmallVector<int64_t, 4> data;
    data.reserve(vals.size());
    for (auto v : vals) {
        data.push_back(static_cast<int64_t>(v));
    }
    return mlir::DenseIntElementsAttr::get(type, data);
}

mlir::Value pad_input(mlir::OpBuilder& b,
                      mlir::Location loc,
                      mlir::Value input,
                      const ov::CoordinateDiff& pads_begin,
                      const ov::CoordinateDiff& pads_end) {
    if ((pads_begin[0] == 0 && pads_begin[1] == 0) &&
        (pads_end[0] == 0 && pads_end[1] == 0)) {
        return input;
    }

    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto elem_ty = input_type.getElementType();
    auto in_shape = input_type.getShape();
    mlir::SmallVector<int64_t, 4> padded_shape(in_shape.begin(), in_shape.end());
    if (padded_shape.size() >= 4) {
        if (padded_shape[2] != mlir::ShapedType::kDynamic) {
            padded_shape[2] += pads_begin[0] + pads_end[0];
        }
        if (padded_shape[3] != mlir::ShapedType::kDynamic) {
            padded_shape[3] += pads_begin[1] + pads_end[1];
        }
    }
    auto padded_type = mlir::RankedTensorType::get(padded_shape, elem_ty);

    mlir::SmallVector<mlir::OpFoldResult, 4> low;
    mlir::SmallVector<mlir::OpFoldResult, 4> high;
    low.reserve(4);
    high.reserve(4);
    low.push_back(b.getI64IntegerAttr(0));
    low.push_back(b.getI64IntegerAttr(0));
    low.push_back(b.getI64IntegerAttr(pads_begin[0]));
    low.push_back(b.getI64IntegerAttr(pads_begin[1]));
    high.push_back(b.getI64IntegerAttr(0));
    high.push_back(b.getI64IntegerAttr(0));
    high.push_back(b.getI64IntegerAttr(pads_end[0]));
    high.push_back(b.getI64IntegerAttr(pads_end[1]));

    auto pad_value = b.create<mlir::arith::ConstantOp>(loc, b.getFloatAttr(elem_ty, 0.0));
    auto pad = b.create<mlir::tensor::PadOp>(loc,
                                             padded_type,
                                             input,
                                             llvm::ArrayRef<mlir::OpFoldResult>(low),
                                             llvm::ArrayRef<mlir::OpFoldResult>(high),
                                             pad_value,
                                             /*nofold=*/false,
                                             mlir::ArrayRef<mlir::NamedAttribute>{});
    return pad.getResult();
}

mlir::Value apply_unary_activation(mlir::OpBuilder& b,
                                   mlir::Location loc,
                                   mlir::Value input,
                                   ActivationKind kind,
                                   float alpha) {
    auto input_type = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!input_type) {
        return input;
    }
    auto elem_ty = input_type.getElementType();
    if (!mlir::isa<mlir::FloatType>(elem_ty)) {
        return input;
    }

    const auto rank = input_type.getRank();
    mlir::SmallVector<int64_t, 4> dims(input_type.getShape().begin(), input_type.getShape().end());
    mlir::SmallVector<mlir::Value, 4> dyn_dims;
    dyn_dims.reserve(rank);
    for (int64_t i = 0; i < rank; ++i) {
        if (dims[i] == mlir::ShapedType::kDynamic) {
            dyn_dims.push_back(b.create<mlir::tensor::DimOp>(loc, input, i));
        }
    }

    auto out_init = b.create<mlir::tensor::EmptyOp>(loc, dims, elem_ty, dyn_dims);
    auto map = mlir::AffineMap::getMultiDimIdentityMap(rank, b.getContext());
    mlir::SmallVector<mlir::AffineMap> maps{map, map};
    mlir::SmallVector<mlir::utils::IteratorType> iters(rank, mlir::utils::IteratorType::parallel);

    auto generic = b.create<mlir::linalg::GenericOp>(
        loc, input_type,
        mlir::ValueRange{input},
        mlir::ValueRange{out_init.getResult()},
        maps,
        iters);
    {
        auto& region = generic.getRegion();
        region.getBlocks().clear();
        auto* block = &region.emplaceBlock();
        block->addArguments({elem_ty, elem_ty}, {loc, loc});
        mlir::OpBuilder body(block, block->begin());
        auto x = block->getArgument(0);
        auto make_float_attr = [&](double v) { return mlir::FloatAttr::get(elem_ty, v); };

        mlir::Value result = x;
        switch (kind) {
            case ActivationKind::Relu: {
                auto zero = body.create<mlir::arith::ConstantOp>(loc, make_float_attr(0.0));
                result = body.create<mlir::arith::MaximumFOp>(loc, x, zero);
                break;
            }
            case ActivationKind::Sigmoid: {
                auto neg = body.create<mlir::arith::NegFOp>(loc, x);
                auto exp = body.create<mlir::math::ExpOp>(loc, neg);
                auto one = body.create<mlir::arith::ConstantOp>(loc, make_float_attr(1.0));
                auto denom = body.create<mlir::arith::AddFOp>(loc, one, exp);
                result = body.create<mlir::arith::DivFOp>(loc, one, denom);
                break;
            }
            case ActivationKind::Tanh: {
                result = body.create<mlir::math::TanhOp>(loc, x);
                break;
            }
            case ActivationKind::Elu: {
                auto alpha_c = body.create<mlir::arith::ConstantOp>(loc, make_float_attr(alpha));
                auto exp = body.create<mlir::math::ExpOp>(loc, x);
                auto one = body.create<mlir::arith::ConstantOp>(loc, make_float_attr(1.0));
                auto expm1 = body.create<mlir::arith::SubFOp>(loc, exp, one);
                auto neg_branch = body.create<mlir::arith::MulFOp>(loc, alpha_c, expm1);
                auto zero = body.create<mlir::arith::ConstantOp>(loc, make_float_attr(0.0));
                auto cond = body.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, x, zero);
                result = body.create<mlir::arith::SelectOp>(loc, cond, x, neg_branch);
                break;
            }
            default:
                break;
        }
        body.create<mlir::linalg::YieldOp>(loc, mlir::ValueRange{result});
    }
    return generic.getResult(0);
}
}  // namespace

mlir::ModuleOp build_mlir_conv2d_from_model(const std::shared_ptr<const ov::Model>& model,
                                            mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                    mlir::tensor::TensorDialect, mlir::arith::ArithDialect,
                    mlir::math::MathDialect, mlir::scf::SCFDialect>();

    std::shared_ptr<const ov::op::v1::Convolution> conv;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto c = ov::as_type_ptr<const ov::op::v1::Convolution>(node)) {
            conv = c;
            break;
        }
    }
    OPENVINO_ASSERT(conv, "Conv2D builder: Convolution op not found");

    const auto in_pshape  = conv->get_input_partial_shape(0);
    const auto w_pshape   = conv->get_input_partial_shape(1);
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

    auto to_elem_ty = [&](ov::element::Type et) -> mlir::Type {
        switch (et) {
            case ov::element::f16: return mlir::Float16Type::get(&ctx);
            case ov::element::f32: return mlir::Float32Type::get(&ctx);
            default: return mlir::Float32Type::get(&ctx);
        }
    };
    auto elem_ty = to_elem_ty(conv->get_output_element_type(0));
    auto make_float_attr = [&](double v) { return mlir::FloatAttr::get(elem_ty, v); };

    const auto in_shape   = to_tensor_shape(in_pshape);
    const auto w_shape    = to_tensor_shape(w_pshape);
    const auto out_shape  = to_tensor_shape(out_pshape);

    auto in_type  = mlir::RankedTensorType::get(in_shape, elem_ty);
    auto w_type   = mlir::RankedTensorType::get(w_shape, elem_ty);
    auto out_type = mlir::RankedTensorType::get(out_shape, elem_ty);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({in_type, w_type}, {out_type});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "conv2d_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    b.setInsertionPointToStart(&func.getBody().front());
    auto loc = mlir::UnknownLoc::get(&ctx);

    auto c0  = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1  = b.create<mlir::arith::ConstantIndexOp>(loc, 1);

    auto input  = func.getArgument(0);
    auto weight = func.getArgument(1);

    auto N     = b.create<mlir::tensor::DimOp>(loc, input, 0);
    auto C_in  = b.create<mlir::tensor::DimOp>(loc, input, 1);
    auto H     = b.create<mlir::tensor::DimOp>(loc, input, 2);
    auto W     = b.create<mlir::tensor::DimOp>(loc, input, 3);
    auto C_out = b.create<mlir::tensor::DimOp>(loc, weight, 0);
    auto kH    = b.create<mlir::tensor::DimOp>(loc, weight, 2);
    auto kW    = b.create<mlir::tensor::DimOp>(loc, weight, 3);

    auto padTop    = b.create<mlir::arith::ConstantIndexOp>(loc, pads_begin[0]);
    auto padLeft   = b.create<mlir::arith::ConstantIndexOp>(loc, pads_begin[1]);
    auto padBottom = b.create<mlir::arith::ConstantIndexOp>(loc, pads_end[0]);
    auto padRight  = b.create<mlir::arith::ConstantIndexOp>(loc, pads_end[1]);
    auto strideH   = b.create<mlir::arith::ConstantIndexOp>(loc, strides[0]);
    auto strideW   = b.create<mlir::arith::ConstantIndexOp>(loc, strides[1]);
    auto dilH      = b.create<mlir::arith::ConstantIndexOp>(loc, dilations[0]);
    auto dilW      = b.create<mlir::arith::ConstantIndexOp>(loc, dilations[1]);

    auto kh_minus_1  = b.create<mlir::arith::SubIOp>(loc, kH, c1);
    auto kw_minus_1  = b.create<mlir::arith::SubIOp>(loc, kW, c1);
    auto eff_filterH = b.create<mlir::arith::MulIOp>(loc, dilH, kh_minus_1);
    auto eff_filterW = b.create<mlir::arith::MulIOp>(loc, dilW, kw_minus_1);

    auto H_pad   = b.create<mlir::arith::AddIOp>(loc, H, padTop);
    auto H_pad2  = b.create<mlir::arith::AddIOp>(loc, H_pad, padBottom);
    auto H_sub   = b.create<mlir::arith::SubIOp>(loc, H_pad2, eff_filterH);
    auto H_sub1  = b.create<mlir::arith::SubIOp>(loc, H_sub, c1);
    auto outHdiv = b.create<mlir::arith::DivSIOp>(loc, H_sub1, strideH);
    auto outH    = b.create<mlir::arith::AddIOp>(loc, outHdiv, c1);

    auto W_pad   = b.create<mlir::arith::AddIOp>(loc, W, padLeft);
    auto W_pad2  = b.create<mlir::arith::AddIOp>(loc, W_pad, padRight);
    auto W_sub   = b.create<mlir::arith::SubIOp>(loc, W_pad2, eff_filterW);
    auto W_sub1  = b.create<mlir::arith::SubIOp>(loc, W_sub, c1);
    auto outWdiv = b.create<mlir::arith::DivSIOp>(loc, W_sub1, strideW);
    auto outW    = b.create<mlir::arith::AddIOp>(loc, outWdiv, c1);

    mlir::SmallVector<mlir::Value> out_dyn;
    const auto shape = out_type.getShape();
    auto push_if_dyn = [&](int idx, mlir::Value v) {
        if (shape[idx] == mlir::ShapedType::kDynamic)
            out_dyn.push_back(v);
    };
    push_if_dyn(0, N);
    push_if_dyn(1, C_out);
    push_if_dyn(2, outH);
    push_if_dyn(3, outW);

    auto out_init = b.create<mlir::tensor::EmptyOp>(loc, out_shape, elem_ty, out_dyn);

    // Build linalg.generic with explicit boundary checks (no tensor.pad, no DenseArrayAttr).
    mlir::AffineExpr n, f, oh, ow, kh, kw, c_dim;
    bindDims(&ctx, n, f, oh, ow, kh, kw, c_dim);
    auto weight_map = mlir::AffineMap::get(7, 0, {f, c_dim, kh, kw}, &ctx);
    auto output_map = mlir::AffineMap::get(7, 0, {n, f, oh, ow}, &ctx);
    mlir::SmallVector<mlir::AffineMap> indexing_maps = {weight_map, output_map};
    mlir::SmallVector<mlir::utils::IteratorType> iter_types = {
        mlir::utils::IteratorType::parallel,  // n
        mlir::utils::IteratorType::parallel,  // f
        mlir::utils::IteratorType::parallel,  // oh
        mlir::utils::IteratorType::parallel,  // ow
        mlir::utils::IteratorType::reduction, // kh
        mlir::utils::IteratorType::reduction, // kw
        mlir::utils::IteratorType::reduction  // c
    };

    auto strideHVal = b.create<mlir::arith::ConstantIndexOp>(loc, strides[0]);
    auto strideWVal = b.create<mlir::arith::ConstantIndexOp>(loc, strides[1]);
    auto dilHVal = b.create<mlir::arith::ConstantIndexOp>(loc, dilations[0]);
    auto dilWVal = b.create<mlir::arith::ConstantIndexOp>(loc, dilations[1]);
    auto padTopVal = b.create<mlir::arith::ConstantIndexOp>(loc, pads_begin[0]);
    auto padLeftVal = b.create<mlir::arith::ConstantIndexOp>(loc, pads_begin[1]);

    auto generic = b.create<mlir::linalg::GenericOp>(
        loc,
        out_type,
        mlir::ValueRange{weight},
        mlir::ValueRange{out_init.getResult()},
        indexing_maps,
        iter_types);

    {
        auto& region = generic.getRegion();
        region.getBlocks().clear();
        auto* block = &region.emplaceBlock();
        block->addArguments({elem_ty, elem_ty}, {loc, loc});  // args: weightVal, acc

        mlir::OpBuilder body(block, block->begin());
        // Indices.
        auto idxN = body.create<mlir::linalg::IndexOp>(loc, 0);
        auto idxF = body.create<mlir::linalg::IndexOp>(loc, 1);
        auto idxOH = body.create<mlir::linalg::IndexOp>(loc, 2);
        auto idxOW = body.create<mlir::linalg::IndexOp>(loc, 3);
        auto idxKH = body.create<mlir::linalg::IndexOp>(loc, 4);
        auto idxKW = body.create<mlir::linalg::IndexOp>(loc, 5);
        auto idxC  = body.create<mlir::linalg::IndexOp>(loc, 6);

        mlir::Value ih = body.create<mlir::arith::AddIOp>(loc,
                                                          body.create<mlir::arith::MulIOp>(loc, idxOH, strideHVal),
                                                          body.create<mlir::arith::MulIOp>(loc, idxKH, dilHVal)).getResult();
        ih = body.create<mlir::arith::SubIOp>(loc, ih, padTopVal).getResult();
        mlir::Value iw = body.create<mlir::arith::AddIOp>(loc,
                                                          body.create<mlir::arith::MulIOp>(loc, idxOW, strideWVal),
                                                          body.create<mlir::arith::MulIOp>(loc, idxKW, dilWVal)).getResult();
        iw = body.create<mlir::arith::SubIOp>(loc, iw, padLeftVal).getResult();

        auto zero_index = body.create<mlir::arith::ConstantIndexOp>(loc, 0).getResult();
        auto ih_ge0 = body.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, ih, zero_index).getResult();
        auto iw_ge0 = body.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, iw, zero_index).getResult();
        auto ih_ltH = body.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, ih, H).getResult();
        auto iw_ltW = body.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, iw, W).getResult();
        auto cond = body.create<mlir::arith::AndIOp>(loc, ih_ge0, iw_ge0).getResult();
        cond = body.create<mlir::arith::AndIOp>(loc, cond, ih_ltH).getResult();
        cond = body.create<mlir::arith::AndIOp>(loc, cond, iw_ltW).getResult();

        mlir::Value input_elem = body.create<mlir::arith::ConstantOp>(loc, b.getFloatAttr(elem_ty, 0.0)).getResult();
        {
            auto if_op = body.create<mlir::scf::IfOp>(loc, mlir::TypeRange{elem_ty}, cond, /*withElseRegion=*/true);
            mlir::OpBuilder thenBuilder = if_op.getThenBodyBuilder();
            auto val = thenBuilder.create<mlir::tensor::ExtractOp>(loc, input, mlir::ValueRange{idxN, idxC, ih, iw});
            thenBuilder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{val});

            mlir::OpBuilder elseBuilder = if_op.getElseBodyBuilder();
            auto zero_val = elseBuilder.create<mlir::arith::ConstantOp>(loc, b.getFloatAttr(elem_ty, 0.0)).getResult();
            elseBuilder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{zero_val});
            input_elem = if_op.getResult(0);
        }

        auto weight_val = block->getArgument(0);
        mlir::Value acc = block->getArgument(1);
        mlir::Value prod;
        if (llvm::isa<mlir::FloatType>(elem_ty)) {
            prod = body.create<mlir::arith::MulFOp>(loc, input_elem, weight_val).getResult();
            acc = body.create<mlir::arith::AddFOp>(loc, acc, prod).getResult();
        } else {
            prod = body.create<mlir::arith::MulIOp>(loc, input_elem, weight_val).getResult();
            acc = body.create<mlir::arith::AddIOp>(loc, acc, prod).getResult();
        }
        body.create<mlir::linalg::YieldOp>(loc, acc);
    }

    generic->setAttr("gfx.pad_begin", make_i64_attr(b, pads_begin));
    generic->setAttr("gfx.pad_end", make_i64_attr(b, pads_end));

    b.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{generic.getResult(0)});
    return module;
}

// Fused conv2d + activation inside one tensor function.
mlir::ModuleOp build_mlir_conv2d_from_model(const std::shared_ptr<const ov::Model>& model,
                                            mlir::MLIRContext& ctx,
                                            std::optional<std::pair<ActivationKind, float>> unary_kind) {
    if (!unary_kind.has_value()) {
        return build_mlir_conv2d_from_model(model, ctx);
    }

    auto module = build_mlir_conv2d_from_model(model, ctx);
    auto func = module.lookupSymbol<mlir::func::FuncOp>("conv2d_main");
    OPENVINO_ASSERT(func, "conv2d_main not found in fused conv builder");
    auto& block = func.getBody().front();
    auto* ret = block.getTerminator();
    auto loc = ret->getLoc();
    mlir::OpBuilder b(ret);
    auto result = ret->getOperand(0);
    auto elem_ty = mlir::cast<mlir::ShapedType>(result.getType()).getElementType();
    auto make_float_attr = [&](double v) { return mlir::FloatAttr::get(elem_ty, v); };

    auto kind = unary_kind->first;
    float alpha = unary_kind->second;
    result = apply_unary_activation(b, loc, result, kind, alpha);
    ret->setOperand(0, result);
    return module;
}

// Conv2D + bias (Add) + optional activation fused into one tensor function.
mlir::ModuleOp build_mlir_conv2d_with_bias_from_model(const std::shared_ptr<const ov::Model>& model,
                                                      mlir::MLIRContext& ctx,
                                                      std::optional<std::pair<ActivationKind, float>> unary_kind) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                    mlir::tensor::TensorDialect, mlir::arith::ArithDialect,
                    mlir::math::MathDialect, mlir::scf::SCFDialect>();

    std::shared_ptr<const ov::op::v1::Convolution> conv;
    std::shared_ptr<const ov::op::v1::Add> add;
    for (const auto& node : model->get_ordered_ops()) {
        if (!conv) {
            if (auto c = ov::as_type_ptr<const ov::op::v1::Convolution>(node)) conv = c;
        }
        if (auto a = ov::as_type_ptr<const ov::op::v1::Add>(node)) add = a;
    }
    OPENVINO_ASSERT(conv && add, "Conv2D+bias builder: Conv or Add not found");

    ov::Output<ov::Node> bias_in;
    if (add->input_value(0).get_node_shared_ptr() == conv) {
        bias_in = add->input_value(1);
    } else if (add->input_value(1).get_node_shared_ptr() == conv) {
        bias_in = add->input_value(0);
    } else {
        OPENVINO_THROW("Add does not consume Conv output");
    }

    const auto in_pshape  = conv->get_input_partial_shape(0);
    const auto w_pshape   = conv->get_input_partial_shape(1);
    const auto b_pshape   = bias_in.get_partial_shape();
    const auto out_pshape = add->get_output_partial_shape(0);

    OPENVINO_ASSERT(in_pshape.rank().is_static() && in_pshape.rank().get_length() == 4, "Conv2D expects rank-4 input");
    OPENVINO_ASSERT(w_pshape.rank().is_static() && w_pshape.rank().get_length() == 4, "Conv2D expects rank-4 weights");
    OPENVINO_ASSERT(out_pshape.rank().is_static() && out_pshape.rank().get_length() == 4, "Conv2D expects rank-4 output");

    auto to_elem_ty = [&](ov::element::Type et) -> mlir::Type {
        switch (et) {
            case ov::element::f16: return mlir::Float16Type::get(&ctx);
            case ov::element::f32: return mlir::Float32Type::get(&ctx);
            default: return mlir::Float32Type::get(&ctx);
        }
    };
    auto elem_ty = to_elem_ty(conv->get_output_element_type(0));
    auto make_float_attr = [&](double v) { return mlir::FloatAttr::get(elem_ty, v); };
    const auto in_shape   = to_tensor_shape(in_pshape);
    const auto w_shape    = to_tensor_shape(w_pshape);
    const auto b_shape    = to_tensor_shape(b_pshape);
    const auto out_shape  = to_tensor_shape(out_pshape);

    auto in_type  = mlir::RankedTensorType::get(in_shape, elem_ty);
    auto w_type   = mlir::RankedTensorType::get(w_shape, elem_ty);
    auto b_type   = mlir::RankedTensorType::get(b_shape, elem_ty);
    auto out_type = mlir::RankedTensorType::get(out_shape, elem_ty);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({in_type, w_type, b_type}, {out_type});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "conv2d_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    b.setInsertionPointToStart(&func.getBody().front());
    auto loc = mlir::UnknownLoc::get(&ctx);

    auto c1  = b.create<mlir::arith::ConstantIndexOp>(loc, 1);

    auto input  = func.getArgument(0);
    auto weight = func.getArgument(1);
    auto bias   = func.getArgument(2);

    auto N     = b.create<mlir::tensor::DimOp>(loc, input, 0);
    auto C_out = b.create<mlir::tensor::DimOp>(loc, weight, 0);
    auto H     = b.create<mlir::tensor::DimOp>(loc, input, 2);
    auto W     = b.create<mlir::tensor::DimOp>(loc, input, 3);
    auto kH    = b.create<mlir::tensor::DimOp>(loc, weight, 2);
    auto kW    = b.create<mlir::tensor::DimOp>(loc, weight, 3);

    const auto pads_begin = conv->get_pads_begin();
    const auto pads_end   = conv->get_pads_end();
    const auto strides    = conv->get_strides();
    const auto dilations  = conv->get_dilations();

    auto padTop    = b.create<mlir::arith::ConstantIndexOp>(loc, pads_begin[0]);
    auto padLeft   = b.create<mlir::arith::ConstantIndexOp>(loc, pads_begin[1]);
    auto padBottom = b.create<mlir::arith::ConstantIndexOp>(loc, pads_end[0]);
    auto padRight  = b.create<mlir::arith::ConstantIndexOp>(loc, pads_end[1]);
    auto strideH   = b.create<mlir::arith::ConstantIndexOp>(loc, strides[0]);
    auto strideW   = b.create<mlir::arith::ConstantIndexOp>(loc, strides[1]);
    auto dilH      = b.create<mlir::arith::ConstantIndexOp>(loc, dilations[0]);
    auto dilW      = b.create<mlir::arith::ConstantIndexOp>(loc, dilations[1]);

    auto kh_minus_1  = b.create<mlir::arith::SubIOp>(loc, kH, c1);
    auto kw_minus_1  = b.create<mlir::arith::SubIOp>(loc, kW, c1);
    auto eff_filterH = b.create<mlir::arith::MulIOp>(loc, dilH, kh_minus_1);
    auto eff_filterW = b.create<mlir::arith::MulIOp>(loc, dilW, kw_minus_1);

    auto H_pad   = b.create<mlir::arith::AddIOp>(loc, H, padTop);
    auto H_pad2  = b.create<mlir::arith::AddIOp>(loc, H_pad, padBottom);
    auto H_sub   = b.create<mlir::arith::SubIOp>(loc, H_pad2, eff_filterH);
    auto H_sub1  = b.create<mlir::arith::SubIOp>(loc, H_sub, c1);
    auto outHdiv = b.create<mlir::arith::DivSIOp>(loc, H_sub1, strideH);
    auto outH    = b.create<mlir::arith::AddIOp>(loc, outHdiv, c1);

    auto W_pad   = b.create<mlir::arith::AddIOp>(loc, W, padLeft);
    auto W_pad2  = b.create<mlir::arith::AddIOp>(loc, W_pad, padRight);
    auto W_sub   = b.create<mlir::arith::SubIOp>(loc, W_pad2, eff_filterW);
    auto W_sub1  = b.create<mlir::arith::SubIOp>(loc, W_sub, c1);
    auto outWdiv = b.create<mlir::arith::DivSIOp>(loc, W_sub1, strideW);
    auto outW    = b.create<mlir::arith::AddIOp>(loc, outWdiv, c1);

    mlir::SmallVector<mlir::Value> out_dyn;
    const auto shape = out_type.getShape();
    auto push_if_dyn = [&](int idx, mlir::Value v) {
        if (shape[idx] == mlir::ShapedType::kDynamic)
            out_dyn.push_back(v);
    };
    push_if_dyn(0, N);
    push_if_dyn(1, C_out);
    push_if_dyn(2, outH);
    push_if_dyn(3, outW);

    auto out_init = b.create<mlir::tensor::EmptyOp>(loc, out_shape, elem_ty, out_dyn);
    auto zero = b.create<mlir::arith::ConstantOp>(loc, b.getFloatAttr(elem_ty, 0.0));
    auto out_filled = b.create<mlir::linalg::FillOp>(loc,
                                                     mlir::ValueRange{zero},
                                                     mlir::ValueRange{out_init.getResult()});

    auto padded = pad_input(b, loc, input, pads_begin, pads_end);
    auto strides_attr = make_i64_attr(b, strides);
    auto dil_attr = make_i64_attr(b, dilations);
    auto conv_op = b.create<mlir::linalg::Conv2DNchwFchwOp>(loc,
                                                            out_type,
                                                            mlir::ValueRange{padded, weight},
                                                            mlir::ValueRange{out_filled.getResult(0)},
                                                            strides_attr,
                                                            dil_attr);
    conv_op->setAttr("gfx.pad_begin", make_i64_attr(b, pads_begin));
    conv_op->setAttr("gfx.pad_end", make_i64_attr(b, pads_end));

    // linalg.generic to add bias with broadcasting over N,H,W when bias is 1D (C_out).
    mlir::AffineMap outMap = mlir::AffineMap::get(4, 0,
                                                  {b.getAffineDimExpr(0), b.getAffineDimExpr(1),
                                                   b.getAffineDimExpr(2), b.getAffineDimExpr(3)}, &ctx);
    mlir::AffineMap biasMap;
    if (b_shape.size() == 1) {
        biasMap = mlir::AffineMap::get(4, 0, {b.getAffineDimExpr(1)}, &ctx);
    } else {
        biasMap = outMap;
    }
    mlir::SmallVector<mlir::AffineMap> affineMaps{outMap, biasMap, outMap};
    mlir::SmallVector<mlir::utils::IteratorType> iters(4, mlir::utils::IteratorType::parallel);
    auto bias_add = b.create<mlir::linalg::GenericOp>(
        loc, mlir::TypeRange{out_type},
        mlir::ValueRange{conv_op.getResult(0), bias},
        mlir::ValueRange{out_init},
        affineMaps,
        iters);
    {
        auto& region = bias_add.getRegion();
        region.getBlocks().clear();  // ensure no default block with missing output arg
        auto* block = &region.emplaceBlock();
        // linalg.generic expects one block argument per input and per output (init tensor).
        block->addArguments({out_type.getElementType(),  // conv result
                             out_type.getElementType(),  // bias
                             out_type.getElementType()}, // output/init
                            {loc, loc, loc});
        mlir::OpBuilder bb(block, block->begin());
        auto sum = bb.create<mlir::arith::AddFOp>(loc, block->getArgument(0), block->getArgument(1));
        bb.create<mlir::linalg::YieldOp>(loc, mlir::ValueRange{sum});
    }

    mlir::Value result = bias_add.getResult(0);

    if (unary_kind.has_value()) {
        auto kind = unary_kind->first;
        float alpha = unary_kind->second;
        result = apply_unary_activation(b, loc, result, kind, alpha);
    }

    b.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{result});
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
