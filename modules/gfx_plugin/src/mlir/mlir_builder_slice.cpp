// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/slice.hpp"

#include <numeric>

namespace ov {
namespace gfx_plugin {

namespace {
mlir::Type to_mlir_type(ov::element::Type et, mlir::MLIRContext& ctx) {
    switch (et) {
        case ov::element::f32: return mlir::Float32Type::get(&ctx);
        case ov::element::f16: return mlir::Float16Type::get(&ctx);
        case ov::element::i8:
        case ov::element::u8:  return mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::Signed);
        case ov::element::i32: return mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Signed);
        case ov::element::i64: return mlir::IntegerType::get(&ctx, 64, mlir::IntegerType::Signed);
        default: OPENVINO_THROW("Slice MLIR: unsupported element type");
    }
}

mlir::SmallVector<int64_t> to_shape(const ov::PartialShape& ps) {
    mlir::SmallVector<int64_t> dims;
    dims.reserve(ps.rank().get_length());
    for (const auto& d : ps) {
        dims.push_back(d.is_dynamic() ? mlir::ShapedType::kDynamic
                                      : static_cast<int64_t>(d.get_length()));
    }
    return dims;
}

std::vector<int64_t> get_const_i64(const std::shared_ptr<const ov::Node>& n) {
    auto c = ov::as_type_ptr<const ov::op::v0::Constant>(n);
    OPENVINO_ASSERT(c, "Slice MLIR: inputs must be Constant");
    return c->cast_vector<int64_t>();
}
}  // namespace

mlir::ModuleOp build_mlir_slice_from_model(const std::shared_ptr<const ov::Model>& model,
                                           mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                    mlir::arith::ArithDialect>();

    std::shared_ptr<const ov::op::v8::Slice> sl;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto s = ov::as_type_ptr<const ov::op::v8::Slice>(node)) {
            OPENVINO_ASSERT(!sl, "Slice MLIR builder: expected single Slice");
            sl = s;
        }
    }
    OPENVINO_ASSERT(sl, "Slice MLIR builder: Slice op not found");

    const auto rank = sl->get_input_shape(0).size();
    auto in_shape = to_shape(sl->get_input_partial_shape(0));
    auto out_shape = to_shape(sl->get_output_partial_shape(0));

    auto elem_ty = to_mlir_type(sl->get_output_element_type(0), ctx);
    auto in_tensor_ty = mlir::RankedTensorType::get(in_shape, elem_ty);
    auto out_tensor_ty = mlir::RankedTensorType::get(out_shape, elem_ty);

    auto starts = get_const_i64(sl->get_input_node_shared_ptr(1));
    auto ends = get_const_i64(sl->get_input_node_shared_ptr(2));
    auto steps = get_const_i64(sl->get_input_node_shared_ptr(3));
    std::vector<int64_t> axes;
    if (sl->get_input_size() > 4) {
        axes = get_const_i64(sl->get_input_node_shared_ptr(4));
    } else {
        axes.resize(starts.size());
        std::iota(axes.begin(), axes.end(), 0);
    }
    OPENVINO_ASSERT(starts.size() == ends.size() && starts.size() == steps.size() &&
                        starts.size() == axes.size(),
                    "Slice MLIR: starts/ends/steps/axes size mismatch");

    std::vector<int64_t> starts_full(rank, 0);
    std::vector<int64_t> steps_full(rank, 1);
    for (size_t i = 0; i < axes.size(); ++i) {
        int64_t axis = axes[i];
        if (axis < 0)
            axis += static_cast<int64_t>(rank);
        OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < rank, "Slice MLIR: axis out of range");
        OPENVINO_ASSERT(steps[i] > 0, "Slice MLIR: only positive steps supported");
        starts_full[static_cast<size_t>(axis)] = starts[i];
        steps_full[static_cast<size_t>(axis)] = steps[i];
    }

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({in_tensor_ty}, {out_tensor_ty});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "slice_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());
    auto empty = b.create<mlir::tensor::EmptyOp>(loc, out_shape, elem_ty);

    llvm::SmallVector<mlir::AffineExpr> in_exprs;
    in_exprs.reserve(rank);
    for (size_t i = 0; i < rank; ++i) {
        auto d = mlir::getAffineDimExpr(static_cast<unsigned>(i), &ctx);
        int64_t step = steps_full[i];
        int64_t start = starts_full[i];
        mlir::AffineExpr expr = d;
        if (step != 1) {
            expr = mlir::getAffineConstantExpr(step, &ctx) * expr;
        }
        if (start != 0) {
            expr = expr + mlir::getAffineConstantExpr(start, &ctx);
        }
        in_exprs.push_back(expr);
    }

    auto map_in = mlir::AffineMap::get(static_cast<unsigned>(rank), 0, in_exprs, &ctx);
    auto map_out = mlir::AffineMap::getMultiDimIdentityMap(static_cast<unsigned>(rank), &ctx);
    llvm::SmallVector<mlir::utils::IteratorType> iters(rank, mlir::utils::IteratorType::parallel);

    auto generic = b.create<mlir::linalg::GenericOp>(
        loc,
        out_tensor_ty,
        mlir::ValueRange{func.getArgument(0)},
        mlir::ValueRange{empty},
        mlir::ArrayRef<mlir::AffineMap>{map_in, map_out},
        mlir::ArrayRef<mlir::utils::IteratorType>(iters));
    {
        auto& region = generic.getRegion();
        region.getBlocks().clear();
        auto* block = &region.emplaceBlock();
        block->addArguments({elem_ty, elem_ty}, {loc, loc});
        mlir::OpBuilder body(block, block->begin());
        body.create<mlir::linalg::YieldOp>(loc, block->getArgument(0));
    }

    b.create<mlir::func::ReturnOp>(loc, generic.getResults());
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
