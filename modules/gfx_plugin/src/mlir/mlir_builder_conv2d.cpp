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
std::vector<int64_t> invert_permutation(const std::vector<int64_t>& permutation) {
    std::vector<int64_t> inverse(permutation.size(), 0);
    for (size_t axis = 0; axis < permutation.size(); ++axis) {
        const auto mapped_axis = static_cast<size_t>(permutation[axis]);
        OPENVINO_ASSERT(mapped_axis < permutation.size(), "Conv2D builder: invalid permutation axis");
        inverse[mapped_axis] = static_cast<int64_t>(axis);
    }
    return inverse;
}

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

mlir::Type to_elem_ty(ov::element::Type et, mlir::MLIRContext& ctx) {
    switch (et) {
        case ov::element::f16: return mlir::Float16Type::get(&ctx);
        case ov::element::f32: return mlir::Float32Type::get(&ctx);
        default: return mlir::Float32Type::get(&ctx);
    }
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

mlir::ModuleOp build_mlir_conv2d_from_node(const std::shared_ptr<const ov::op::v1::Convolution>& conv,
                                           mlir::MLIRContext& ctx,
                                           const MlirInputTransformDesc* input_transform) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                    mlir::tensor::TensorDialect, mlir::arith::ArithDialect,
                    mlir::math::MathDialect, mlir::scf::SCFDialect>();

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

    const auto pads_begin = conv->get_pads_begin();
    const auto pads_end = conv->get_pads_end();
    const auto strides = conv->get_strides();
    const auto dilations = conv->get_dilations();
    const auto weight_shape = conv->get_input_shape(1);
    const ov::Strides unit_strides{1, 1};
    const ov::CoordinateDiff zero_pads{0, 0};
    OPENVINO_ASSERT(weight_shape.size() == 4, "Conv2D builder expects rank-4 weights");

    const bool has_input_transform = input_transform && input_transform->has_transpose();
    if (has_input_transform) {
        OPENVINO_ASSERT(strides == unit_strides,
                        "Transformed Conv2D builder currently supports only stride-1 pointwise conv");
        OPENVINO_ASSERT(dilations == unit_strides,
                        "Transformed Conv2D builder currently supports only dilation-1 pointwise conv");
        OPENVINO_ASSERT(pads_begin == zero_pads && pads_end == zero_pads,
                        "Transformed Conv2D builder currently supports only pad-0 pointwise conv");
        OPENVINO_ASSERT(weight_shape[2] == 1 && weight_shape[3] == 1,
                        "Transformed Conv2D builder currently supports only 1x1 conv");
        OPENVINO_ASSERT(input_transform->transpose_permutation.size() == 4,
                        "Transformed Conv2D builder expects rank-4 permutation");
    }

    const auto elem_ty = to_elem_ty(conv->get_output_element_type(0), ctx);
    const auto logical_in_shape = to_tensor_shape(in_pshape);
    const auto source_in_shape =
        has_input_transform ? mlir::SmallVector<int64_t>(input_transform->source_shape.begin(), input_transform->source_shape.end())
                            : logical_in_shape;
    const auto w_shape = to_tensor_shape(w_pshape);
    const auto out_shape = to_tensor_shape(out_pshape);

    auto in_type = mlir::RankedTensorType::get(source_in_shape, elem_ty);
    auto w_type = mlir::RankedTensorType::get(w_shape, elem_ty);
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

    auto input = func.getArgument(0);
    auto weight = func.getArgument(1);

    if (!has_input_transform) {
        auto padded = pad_input(b, loc, input, pads_begin, pads_end);
        auto out_init = b.create<mlir::tensor::EmptyOp>(loc, out_shape, elem_ty);
        auto zero = b.create<mlir::arith::ConstantOp>(loc, b.getFloatAttr(elem_ty, 0.0));
        auto out_filled = b.create<mlir::linalg::FillOp>(loc,
                                                         mlir::ValueRange{zero},
                                                         mlir::ValueRange{out_init.getResult()});
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
        b.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{conv_op.getResult(0)});
        return module;
    }

    const auto inverse_permutation = invert_permutation(input_transform->transpose_permutation);
    auto out_init = b.create<mlir::tensor::EmptyOp>(loc, out_shape, elem_ty);
    auto zero = b.create<mlir::arith::ConstantOp>(loc, b.getFloatAttr(elem_ty, 0.0));
    auto out_filled = b.create<mlir::linalg::FillOp>(loc,
                                                     mlir::ValueRange{zero},
                                                     mlir::ValueRange{out_init.getResult()});

    auto n_expr = mlir::getAffineDimExpr(0, &ctx);
    auto co_expr = mlir::getAffineDimExpr(1, &ctx);
    auto h_expr = mlir::getAffineDimExpr(2, &ctx);
    auto w_expr = mlir::getAffineDimExpr(3, &ctx);
    auto ci_expr = mlir::getAffineDimExpr(4, &ctx);
    auto zero_expr = mlir::getAffineConstantExpr(0, &ctx);
    llvm::SmallVector<mlir::AffineExpr> logical_source_indices{n_expr, ci_expr, h_expr, w_expr};
    llvm::SmallVector<mlir::AffineExpr> source_indices;
    source_indices.reserve(inverse_permutation.size());
    for (int64_t axis : inverse_permutation) {
        OPENVINO_ASSERT(axis >= 0 && axis < static_cast<int64_t>(logical_source_indices.size()),
                        "Conv2D builder: invalid inverse permutation");
        source_indices.push_back(logical_source_indices[static_cast<size_t>(axis)]);
    }

    auto source_map = mlir::AffineMap::get(5, 0, source_indices, &ctx);
    auto weight_map = mlir::AffineMap::get(5, 0, {co_expr, ci_expr, zero_expr, zero_expr}, &ctx);
    auto out_map = mlir::AffineMap::get(5, 0, {n_expr, co_expr, h_expr, w_expr}, &ctx);

    llvm::SmallVector<mlir::utils::IteratorType> iterators = {
        mlir::utils::IteratorType::parallel,
        mlir::utils::IteratorType::parallel,
        mlir::utils::IteratorType::parallel,
        mlir::utils::IteratorType::parallel,
        mlir::utils::IteratorType::reduction
    };

    auto generic = b.create<mlir::linalg::GenericOp>(
        loc,
        out_type,
        mlir::ValueRange{input, weight},
        mlir::ValueRange{out_filled.getResult(0)},
        mlir::ArrayRef<mlir::AffineMap>{source_map, weight_map, out_map},
        mlir::ArrayRef<mlir::utils::IteratorType>(iterators));
    {
        auto& region = generic.getRegion();
        region.getBlocks().clear();
        auto* block = &region.emplaceBlock();
        block->addArguments({elem_ty, elem_ty, elem_ty}, {loc, loc, loc});
        mlir::OpBuilder body(block, block->begin());
        auto lhs = block->getArgument(0);
        auto rhs = block->getArgument(1);
        auto acc = block->getArgument(2);
        auto mul = body.create<mlir::arith::MulFOp>(loc, lhs, rhs);
        auto sum = body.create<mlir::arith::AddFOp>(loc, acc, mul);
        body.create<mlir::linalg::YieldOp>(loc, mlir::ValueRange{sum.getResult()});
    }

    b.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{generic.getResult(0)});
    return module;
}

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

    auto elem_ty = to_elem_ty(conv->get_output_element_type(0), ctx);
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

    auto c1  = b.create<mlir::arith::ConstantIndexOp>(loc, 1);

    auto input  = func.getArgument(0);
    auto weight = func.getArgument(1);

    auto N     = b.create<mlir::tensor::DimOp>(loc, input, 0);
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

    b.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{conv_op.getResult(0)});
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
