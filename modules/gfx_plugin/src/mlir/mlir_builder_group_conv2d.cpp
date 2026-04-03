// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "openvino/core/model.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/strides.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "runtime/gfx_logger.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
std::vector<int64_t> invert_permutation(const std::vector<int64_t>& permutation) {
    std::vector<int64_t> inverse(permutation.size(), -1);
    for (size_t i = 0; i < permutation.size(); ++i) {
        const auto axis = permutation[i];
        OPENVINO_ASSERT(axis >= 0 && axis < static_cast<int64_t>(permutation.size()),
                        "GroupConv2D builder: permutation axis out of range");
        OPENVINO_ASSERT(inverse[static_cast<size_t>(axis)] < 0,
                        "GroupConv2D builder: permutation axis repeated");
        inverse[static_cast<size_t>(axis)] = static_cast<int64_t>(i);
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

mlir::Value pad_input(mlir::OpBuilder& b,
                      mlir::Location loc,
                      mlir::Value input,
                      const ov::CoordinateDiff& pads_begin,
                      const ov::CoordinateDiff& pads_end,
                      const MlirInputTransformDesc* input_transform = nullptr) {
    if ((pads_begin[0] == 0 && pads_begin[1] == 0) &&
        (pads_end[0] == 0 && pads_end[1] == 0)) {
        return input;
    }

    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto elem_ty = input_type.getElementType();
    auto in_shape = input_type.getShape();
    mlir::SmallVector<int64_t, 4> padded_shape(in_shape.begin(), in_shape.end());
    OPENVINO_ASSERT(padded_shape.size() >= 4, "GroupConv2D builder: rank-4 input expected for padding");
    const size_t pad_h_axis = (input_transform && input_transform->has_transpose())
                                  ? static_cast<size_t>(input_transform->transpose_permutation.at(2))
                                  : 2u;
    const size_t pad_w_axis = (input_transform && input_transform->has_transpose())
                                  ? static_cast<size_t>(input_transform->transpose_permutation.at(3))
                                  : 3u;
    OPENVINO_ASSERT(pad_h_axis < padded_shape.size() && pad_w_axis < padded_shape.size(),
                    "GroupConv2D builder: pad axes out of range");
    if (padded_shape[pad_h_axis] != mlir::ShapedType::kDynamic) {
        padded_shape[pad_h_axis] += pads_begin[0] + pads_end[0];
    }
    if (padded_shape[pad_w_axis] != mlir::ShapedType::kDynamic) {
        padded_shape[pad_w_axis] += pads_begin[1] + pads_end[1];
    }
    auto padded_type = mlir::RankedTensorType::get(padded_shape, elem_ty);

    mlir::SmallVector<mlir::OpFoldResult, 4> low;
    mlir::SmallVector<mlir::OpFoldResult, 4> high;
    low.reserve(4);
    high.reserve(4);
    for (size_t i = 0; i < padded_shape.size(); ++i) {
        const int64_t low_pad = (i == pad_h_axis) ? pads_begin[0] : ((i == pad_w_axis) ? pads_begin[1] : 0);
        const int64_t high_pad = (i == pad_h_axis) ? pads_end[0] : ((i == pad_w_axis) ? pads_end[1] : 0);
        low.push_back(b.getI64IntegerAttr(low_pad));
        high.push_back(b.getI64IntegerAttr(high_pad));
    }

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
}  // namespace

mlir::ModuleOp build_mlir_group_conv2d_from_model(const std::shared_ptr<const ov::Model>& model,
                                                  mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                    mlir::tensor::TensorDialect, mlir::arith::ArithDialect,
                    mlir::scf::SCFDialect>();

    std::shared_ptr<const ov::op::v1::GroupConvolution> gconv;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto c = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)) {
            gconv = c;
            break;
        }
    }
    OPENVINO_ASSERT(gconv, "GroupConv2D builder: GroupConvolution op not found");
    return build_mlir_group_conv2d_from_node(gconv, ctx, nullptr);
}

mlir::ModuleOp build_mlir_group_conv2d_from_node(const std::shared_ptr<const ov::op::v1::GroupConvolution>& gconv,
                                                 mlir::MLIRContext& ctx,
                                                 const MlirInputTransformDesc* input_transform) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                    mlir::tensor::TensorDialect, mlir::arith::ArithDialect,
                    mlir::scf::SCFDialect>();

    OPENVINO_ASSERT(gconv, "GroupConv2D builder: GroupConvolution op is null");

    const auto in_pshape  = gconv->get_input_partial_shape(0);
    const auto w_pshape   = gconv->get_input_partial_shape(1);
    const auto out_pshape = gconv->get_output_partial_shape(0);
    OPENVINO_ASSERT(in_pshape.rank().is_static() && in_pshape.rank().get_length() == 4,
                    "GroupConv2D builder expects rank-4 input");
    OPENVINO_ASSERT(w_pshape.rank().is_static() && w_pshape.rank().get_length() == 5,
                    "GroupConv2D builder expects rank-5 weights");
    OPENVINO_ASSERT(out_pshape.rank().is_static() && out_pshape.rank().get_length() == 4,
                    "GroupConv2D builder expects rank-4 output");

    const bool has_input_transform = input_transform && input_transform->has_transpose();
    const auto logical_in_shape = to_tensor_shape(in_pshape);
    mlir::SmallVector<int64_t> in_shape =
        has_input_transform
            ? mlir::SmallVector<int64_t>(input_transform->source_shape.begin(), input_transform->source_shape.end())
            : logical_in_shape;
    const auto w_shape    = to_tensor_shape(w_pshape);
    const auto out_shape  = to_tensor_shape(out_pshape);
    const auto pads_begin = gconv->get_pads_begin();
    const auto pads_end   = gconv->get_pads_end();
    const auto strides    = gconv->get_strides();
    const auto dilations  = gconv->get_dilations();
    if (gfx_log_debug_enabled()) {
        gfx_log_debug("MLIR") << "GroupConv2D in=" << in_pshape << " w=" << w_pshape << " out=" << out_pshape
                              << " transformed_input=" << (has_input_transform ? "yes" : "no")
                                         << " pads_begin=(" << pads_begin[0] << "," << pads_begin[1] << ")"
                                         << " pads_end=(" << pads_end[0] << "," << pads_end[1] << ")"
                                         << " strides=(" << strides[0] << "," << strides[1] << ")"
                                         << " dilations=(" << dilations[0] << "," << dilations[1] << ")";
    }
    OPENVINO_ASSERT(w_shape[0] != mlir::ShapedType::kDynamic,
                    "GroupConv2D builder: group dimension must be static");
    const auto groups = static_cast<size_t>(w_shape[0]);

    auto to_elem_ty = [&](ov::element::Type et) -> mlir::Type {
        switch (et) {
            case ov::element::f16: return mlir::Float16Type::get(&ctx);
            case ov::element::f32: return mlir::Float32Type::get(&ctx);
            default: return mlir::Float32Type::get(&ctx);
        }
    };
    auto elem_ty = to_elem_ty(gconv->get_output_element_type(0));
    auto in_type  = mlir::RankedTensorType::get(in_shape, elem_ty);
    auto w_type   = mlir::RankedTensorType::get(w_shape, elem_ty);
    auto out_type = mlir::RankedTensorType::get(out_shape, elem_ty);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({in_type, w_type}, {out_type});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "group_conv2d_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    b.setInsertionPointToStart(&func.getBody().front());
    auto loc = mlir::UnknownLoc::get(&ctx);

    auto input  = func.getArgument(0);
    auto weight = func.getArgument(1);

    const int64_t in_c = logical_in_shape[1];
    const int64_t out_c = out_shape[1];
    const int64_t group_dim = w_shape[0];
    OPENVINO_ASSERT(group_dim == static_cast<int64_t>(groups),
                    "GroupConv2D builder: group dimension mismatch");

    if (groups == 1) {
        OPENVINO_ASSERT(!has_input_transform,
                        "GroupConv2D builder: transformed input for groups==1 is not supported");
        auto padded = pad_input(b, loc, input, pads_begin, pads_end);
        auto out_init = b.create<mlir::tensor::EmptyOp>(loc, out_shape, elem_ty);
        auto strides_attr = make_i64_attr(b, strides);
        auto dil_attr = make_i64_attr(b, dilations);
        auto conv_op = b.create<mlir::linalg::Conv2DNchwFchwOp>(loc,
                                                                out_type,
                                                                mlir::ValueRange{padded, weight},
                                                                mlir::ValueRange{out_init},
                                                                strides_attr,
                                                                dil_attr);
        conv_op->setAttr("gfx.pad_begin", make_i64_attr(b, pads_begin));
        conv_op->setAttr("gfx.pad_end", make_i64_attr(b, pads_end));
        b.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{conv_op.getResult(0)});
        return module;
    }

    if (gfx_log_debug_enabled()) {
        gfx_log_debug("MLIR") << "GroupConv2D depthwise path: groups=" << groups;
    }

    OPENVINO_ASSERT(in_c != mlir::ShapedType::kDynamic &&
                        out_c != mlir::ShapedType::kDynamic &&
                        in_c == static_cast<int64_t>(groups) &&
                        out_c == static_cast<int64_t>(groups),
                    "GroupConv2D builder: only depthwise (groups == in_channels == out_channels) is supported");

    OPENVINO_ASSERT(w_shape[1] != mlir::ShapedType::kDynamic &&
                        w_shape[2] != mlir::ShapedType::kDynamic &&
                        w_shape[1] == 1 && w_shape[2] == 1,
                    "GroupConv2D builder: depthwise expects weights shape [G,1,1,KH,KW]");

    OPENVINO_ASSERT(mlir::isa<mlir::FloatType>(elem_ty),
                    "GroupConv2D builder: depthwise expects floating point input");

    auto padded = pad_input(b, loc, input, pads_begin, pads_end, input_transform);
    auto out_init = b.create<mlir::tensor::EmptyOp>(loc, out_shape, elem_ty);
    auto zero_val = b.create<mlir::arith::ConstantOp>(loc, b.getZeroAttr(elem_ty));
    auto out_filled = b.create<mlir::linalg::FillOp>(loc,
                                                     mlir::ValueRange{zero_val},
                                                     mlir::ValueRange{out_init.getResult()});

    if (gfx_log_debug_enabled()) {
        gfx_log_debug("MLIR") << "GroupConv2D depthwise linalg.generic";
    }

    auto d0 = b.getAffineDimExpr(0);
    auto d1 = b.getAffineDimExpr(1);
    auto d2 = b.getAffineDimExpr(2);
    auto d3 = b.getAffineDimExpr(3);
    auto d4 = b.getAffineDimExpr(4);
    auto d5 = b.getAffineDimExpr(5);

    auto in_h_expr = d2 * static_cast<int64_t>(strides[0]) +
                     d4 * static_cast<int64_t>(dilations[0]);
    auto in_w_expr = d3 * static_cast<int64_t>(strides[1]) +
                     d5 * static_cast<int64_t>(dilations[1]);

    llvm::SmallVector<mlir::AffineExpr, 4> input_exprs;
    if (has_input_transform) {
        OPENVINO_ASSERT(input_transform->transpose_permutation.size() == logical_in_shape.size(),
                        "GroupConv2D builder: transform permutation rank mismatch");
        const auto inverse_permutation = invert_permutation(input_transform->transpose_permutation);
        input_exprs.reserve(in_shape.size());
        for (size_t src_dim = 0; src_dim < in_shape.size(); ++src_dim) {
            const auto logical_dim = static_cast<size_t>(inverse_permutation[src_dim]);
            switch (logical_dim) {
                case 0:
                    input_exprs.push_back(d0);
                    break;
                case 1:
                    input_exprs.push_back(d1);
                    break;
                case 2:
                    input_exprs.push_back(in_h_expr);
                    break;
                case 3:
                    input_exprs.push_back(in_w_expr);
                    break;
                default:
                    OPENVINO_THROW("GroupConv2D builder: unsupported logical dim ", logical_dim);
            }
        }
    } else {
        input_exprs = {d0, d1, in_h_expr, in_w_expr};
    }
    auto input_map = mlir::AffineMap::get(6, 0, input_exprs, &ctx);
    auto zero = b.getAffineConstantExpr(0);
    auto weight_map = mlir::AffineMap::get(6, 0, {d1, zero, zero, d4, d5}, &ctx);
    auto out_map = mlir::AffineMap::get(6, 0, {d0, d1, d2, d3}, &ctx);

    llvm::SmallVector<mlir::AffineMap, 3> maps{input_map, weight_map, out_map};
    llvm::SmallVector<mlir::utils::IteratorType, 6> iters{
        mlir::utils::IteratorType::parallel,
        mlir::utils::IteratorType::parallel,
        mlir::utils::IteratorType::parallel,
        mlir::utils::IteratorType::parallel,
        mlir::utils::IteratorType::reduction,
        mlir::utils::IteratorType::reduction,
    };

    auto generic = b.create<mlir::linalg::GenericOp>(
        loc,
        out_type,
        mlir::ValueRange{padded, weight},
        mlir::ValueRange{out_filled.getResult(0)},
        maps,
        iters);
    {
        auto& region = generic.getRegion();
        region.getBlocks().clear();
        auto* block = &region.emplaceBlock();
        block->addArguments({elem_ty, elem_ty, elem_ty}, {loc, loc, loc});
        mlir::OpBuilder body(block, block->begin());
        auto in_val = block->getArgument(0);
        auto w_val = block->getArgument(1);
        auto acc = block->getArgument(2);
        auto mul = body.create<mlir::arith::MulFOp>(loc, in_val, w_val);
        auto sum = body.create<mlir::arith::AddFOp>(loc, acc, mul);
        body.create<mlir::linalg::YieldOp>(loc, mlir::ValueRange{sum});
    }

    b.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{generic.getResult(0)});
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
