// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/gfx_mlir_type_utils.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/strided_slice.hpp"

#include <numeric>

namespace ov {
namespace gfx_plugin {

namespace {
std::vector<int64_t> get_const_i64(const ov::Output<ov::Node>& source, const char* what) {
    auto c = ov::util::get_constant_from_source(source);
    OPENVINO_ASSERT(c, "Slice MLIR: ", what, " must be Constant");
    return c->cast_vector<int64_t>();
}

struct SliceSpec {
    std::vector<int64_t> starts_full;
    std::vector<int64_t> steps_full;
};

int64_t normalize_index(int64_t index, int64_t dim, bool is_begin) {
    if (index < 0) {
        index += dim;
    }
    if (is_begin) {
        return std::clamp<int64_t>(index, 0, dim);
    }
    return std::clamp<int64_t>(index, -1, dim);
}

SliceSpec build_slice_spec(const std::shared_ptr<const ov::Node>& node,
                           const ov::Shape& in_shape,
                           const ov::Shape& out_shape) {
    const size_t rank = in_shape.size();
    OPENVINO_ASSERT(rank == out_shape.size(),
                    "Slice MLIR: rank-changing slice is not supported");

    SliceSpec spec;
    spec.starts_full.assign(rank, 0);
    spec.steps_full.assign(rank, 1);

    if (auto sl = ov::as_type_ptr<const ov::op::v8::Slice>(node)) {
        auto starts = get_const_i64(sl->input_value(1), "Slice starts");
        auto ends = get_const_i64(sl->input_value(2), "Slice ends");
        auto steps = get_const_i64(sl->input_value(3), "Slice steps");
        std::vector<int64_t> axes;
        if (sl->get_input_size() > 4) {
            axes = get_const_i64(sl->input_value(4), "Slice axes");
        } else {
            axes.resize(starts.size());
            std::iota(axes.begin(), axes.end(), 0);
        }
        OPENVINO_ASSERT(starts.size() == ends.size() && starts.size() == steps.size() && starts.size() == axes.size(),
                        "Slice MLIR: starts/ends/steps/axes size mismatch");
        for (size_t i = 0; i < axes.size(); ++i) {
            int64_t axis = axes[i];
            if (axis < 0) {
                axis += static_cast<int64_t>(rank);
            }
            OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < rank, "Slice MLIR: axis out of range");
            OPENVINO_ASSERT(steps[i] > 0, "Slice MLIR: only positive steps supported");
            const auto dim = static_cast<int64_t>(in_shape[static_cast<size_t>(axis)]);
            spec.starts_full[static_cast<size_t>(axis)] = normalize_index(starts[i], dim, true);
            spec.steps_full[static_cast<size_t>(axis)] = steps[i];
        }
        return spec;
    }

    auto ss = ov::as_type_ptr<const ov::op::v1::StridedSlice>(node);
    OPENVINO_ASSERT(ss, "Slice MLIR builder: Slice/StridedSlice op not found");
    const auto& begin_mask = ss->get_begin_mask();
    const auto& end_mask = ss->get_end_mask();
    const auto& new_axis_mask = ss->get_new_axis_mask();
    const auto& shrink_axis_mask = ss->get_shrink_axis_mask();
    const auto& ellipsis_mask = ss->get_ellipsis_mask();
    OPENVINO_ASSERT(std::all_of(new_axis_mask.begin(), new_axis_mask.end(), [](int64_t v) { return v == 0; }),
                    "Slice MLIR: StridedSlice new_axis_mask is not supported");
    OPENVINO_ASSERT(std::all_of(shrink_axis_mask.begin(), shrink_axis_mask.end(), [](int64_t v) { return v == 0; }),
                    "Slice MLIR: StridedSlice shrink_axis_mask is not supported");
    OPENVINO_ASSERT(std::all_of(ellipsis_mask.begin(), ellipsis_mask.end(), [](int64_t v) { return v == 0; }),
                    "Slice MLIR: StridedSlice ellipsis_mask is not supported");

    auto begin = get_const_i64(ss->input_value(1), "StridedSlice begin");
    auto end = get_const_i64(ss->input_value(2), "StridedSlice end");
    std::vector<int64_t> strides(rank, 1);
    if (ss->get_input_size() > 3) {
        auto values = get_const_i64(ss->input_value(3), "StridedSlice strides");
        OPENVINO_ASSERT(values.size() <= rank, "Slice MLIR: StridedSlice strides rank mismatch");
        std::copy(values.begin(), values.end(), strides.begin());
    }
    OPENVINO_ASSERT(begin.size() <= rank && end.size() <= rank, "Slice MLIR: StridedSlice begin/end rank mismatch");
    for (size_t axis = 0; axis < rank; ++axis) {
        const auto dim = static_cast<int64_t>(in_shape[axis]);
        const bool masked_begin = axis < begin_mask.size() && begin_mask[axis] != 0;
        const bool masked_end = axis < end_mask.size() && end_mask[axis] != 0;
        const int64_t step = strides[axis];
        OPENVINO_ASSERT(step > 0, "Slice MLIR: StridedSlice only positive steps supported");
        int64_t start = axis < begin.size() ? begin[axis] : 0;
        int64_t finish = axis < end.size() ? end[axis] : dim;
        start = masked_begin ? 0 : normalize_index(start, dim, true);
        finish = masked_end ? dim : normalize_index(finish, dim, false);
        (void)finish;
        spec.starts_full[axis] = start;
        spec.steps_full[axis] = step;
    }
    return spec;
}
}  // namespace

mlir::ModuleOp build_mlir_slice_from_model(const std::shared_ptr<const ov::Model>& model,
                                           mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                    mlir::arith::ArithDialect>();

    std::shared_ptr<const ov::Node> slice_node;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::as_type_ptr<const ov::op::v8::Slice>(node) || ov::as_type_ptr<const ov::op::v1::StridedSlice>(node)) {
            OPENVINO_ASSERT(!slice_node, "Slice MLIR builder: expected single Slice/StridedSlice");
            slice_node = node;
        }
    }
    OPENVINO_ASSERT(slice_node, "Slice MLIR builder: Slice/StridedSlice op not found");

    const auto in_shape_ov = slice_node->get_input_shape(0);
    const auto out_shape_ov = slice_node->get_output_shape(0);
    const auto rank = in_shape_ov.size();
    auto in_shape = to_shape(slice_node->get_input_partial_shape(0));
    auto out_shape = to_shape(slice_node->get_output_partial_shape(0));

    auto elem_ty = to_mlir_type(slice_node->get_output_element_type(0),
                                ctx,
                                /*fallback_f32=*/false,
                                /*allow_unsigned=*/false,
                                /*allow_small_ints=*/true);
    auto in_tensor_ty = mlir::RankedTensorType::get(in_shape, elem_ty);
    auto out_tensor_ty = mlir::RankedTensorType::get(out_shape, elem_ty);
    auto spec = build_slice_spec(slice_node, in_shape_ov, out_shape_ov);

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
        int64_t step = spec.steps_full[i];
        int64_t start = spec.starts_full[i];
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
