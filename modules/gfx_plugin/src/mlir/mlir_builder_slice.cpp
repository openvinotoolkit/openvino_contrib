// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/gfx_mlir_type_utils.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
#include <optional>

namespace ov {
namespace gfx_plugin {

namespace {
std::vector<int64_t> get_const_i64(const ov::Output<ov::Node>& source, const char* what) {
    auto c = ov::util::get_constant_from_source(source);
    OPENVINO_ASSERT(c, "Slice MLIR: ", what, " must be Constant");
    return c->cast_vector<int64_t>();
}

std::optional<std::vector<int64_t>> get_optional_const_i64(const ov::Output<ov::Node>& source) {
    auto c = ov::util::get_constant_from_source(source);
    if (!c) {
        return std::nullopt;
    }
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

int64_t normalize_strided_slice_start(int64_t index, int64_t dim, int64_t step) {
    if (index < 0) {
        index += dim;
    }
    if (step < 0) {
        return std::clamp<int64_t>(index, -1, dim - 1);
    }
    return std::clamp<int64_t>(index, 0, dim);
}

SliceSpec build_slice_spec(const std::shared_ptr<const ov::Node>& node,
                           const ov::PartialShape& in_pshape,
                           const ov::PartialShape& out_pshape) {
    OPENVINO_ASSERT(in_pshape.rank().is_static() && out_pshape.rank().is_static(),
                    "Slice MLIR: input/output ranks must be static");
    const size_t rank = static_cast<size_t>(in_pshape.rank().get_length());
    OPENVINO_ASSERT(rank == static_cast<size_t>(out_pshape.rank().get_length()),
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
            OPENVINO_ASSERT(steps[i] != 0, "Slice MLIR: zero step is not supported");
            if (in_pshape[static_cast<size_t>(axis)].is_dynamic()) {
                if (starts[i] < 0 || steps[i] < 0) {
                    spec.starts_full[static_cast<size_t>(axis)] = 0;
                    spec.steps_full[static_cast<size_t>(axis)] = 1;
                    continue;
                }
                spec.starts_full[static_cast<size_t>(axis)] = starts[i];
            } else {
                const auto dim = static_cast<int64_t>(in_pshape[static_cast<size_t>(axis)].get_length());
                spec.starts_full[static_cast<size_t>(axis)] =
                    normalize_strided_slice_start(starts[i], dim, steps[i]);
            }
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
    auto end_const = get_optional_const_i64(ss->input_value(2));
    std::vector<int64_t> end = end_const.value_or(std::vector<int64_t>{});
    std::vector<int64_t> strides(rank, 1);
    if (ss->get_input_size() > 3) {
        auto values = get_const_i64(ss->input_value(3), "StridedSlice strides");
        OPENVINO_ASSERT(values.size() <= rank, "Slice MLIR: StridedSlice strides rank mismatch");
        std::copy(values.begin(), values.end(), strides.begin());
    }
    OPENVINO_ASSERT(begin.size() <= rank && (!end_const || end.size() <= rank),
                    "Slice MLIR: StridedSlice begin/end rank mismatch");
    for (size_t axis = 0; axis < rank; ++axis) {
        const bool masked_begin = axis < begin_mask.size() && begin_mask[axis] != 0;
        const bool masked_end = axis < end_mask.size() && end_mask[axis] != 0;
        const int64_t step = strides[axis];
        OPENVINO_ASSERT(step != 0, "Slice MLIR: StridedSlice zero step is not supported");
        int64_t start = axis < begin.size() ? begin[axis] : 0;
        if (in_pshape[axis].is_dynamic()) {
            if (step < 0 || (!masked_begin && start < 0) ||
                (end_const && !masked_end && axis < end.size() && end[axis] < 0)) {
                // Dynamic backward slices are resolved by runtime metadata in MlirStage.
                // Keep the carrier MLIR op structurally valid; codegen ignores these
                // placeholder offsets/strides when runtime slice arguments are enabled.
                start = 0;
                spec.steps_full[axis] = 1;
                continue;
            }
            start = masked_begin ? 0 : start;
        } else {
            const auto dim = static_cast<int64_t>(in_pshape[axis].get_length());
            int64_t finish = end_const && axis < end.size() ? end[axis] : dim;
            start = masked_begin ? (step < 0 ? dim - 1 : 0)
                                 : normalize_strided_slice_start(start, dim, step);
            finish = masked_end ? (step < 0 ? -1 : dim) : normalize_index(finish, dim, false);
            (void)finish;
        }
        spec.starts_full[axis] = start;
        spec.steps_full[axis] = step < 0 ? 1 : step;
    }
    return spec;
}
}  // namespace

mlir::ModuleOp build_mlir_slice_from_model(const std::shared_ptr<const ov::Model>& model,
                                           mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();

    std::shared_ptr<const ov::Node> slice_node;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::as_type_ptr<const ov::op::v8::Slice>(node) || ov::as_type_ptr<const ov::op::v1::StridedSlice>(node)) {
            OPENVINO_ASSERT(!slice_node, "Slice MLIR builder: expected single Slice/StridedSlice");
            slice_node = node;
        }
    }
    OPENVINO_ASSERT(slice_node, "Slice MLIR builder: Slice/StridedSlice op not found");

    const auto in_pshape = slice_node->get_input_partial_shape(0);
    const auto out_pshape = slice_node->get_output_partial_shape(0);
    OPENVINO_ASSERT(in_pshape.rank().is_static() && out_pshape.rank().is_static(),
                    "Slice MLIR: input/output ranks must be static");
    const auto rank = static_cast<size_t>(in_pshape.rank().get_length());
    auto in_shape = to_shape(in_pshape);
    auto out_shape = to_shape(out_pshape);

    auto elem_ty = to_mlir_type(slice_node->get_output_element_type(0),
                                ctx,
                                /*fallback_f32=*/false,
                                /*allow_unsigned=*/false,
                                /*allow_small_ints=*/true);
    auto in_tensor_ty = mlir::RankedTensorType::get(in_shape, elem_ty);
    auto out_tensor_ty = mlir::RankedTensorType::get(out_shape, elem_ty);
    auto spec = build_slice_spec(slice_node, in_pshape, out_pshape);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({in_tensor_ty}, {out_tensor_ty});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "slice_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());
    mlir::SmallVector<mlir::OpFoldResult> offsets;
    mlir::SmallVector<mlir::OpFoldResult> sizes;
    mlir::SmallVector<mlir::OpFoldResult> strides;
    offsets.reserve(rank);
    sizes.reserve(rank);
    strides.reserve(rank);
    for (size_t i = 0; i < rank; ++i) {
        offsets.push_back(b.getIndexAttr(spec.starts_full[i]));
        if (out_shape[i] == mlir::ShapedType::kDynamic) {
            sizes.push_back(b.create<mlir::tensor::DimOp>(loc, func.getArgument(0), static_cast<int64_t>(i)).getResult());
        } else {
            sizes.push_back(b.getIndexAttr(out_shape[i]));
        }
        strides.push_back(b.getIndexAttr(spec.steps_full[i]));
    }
    auto slice = b.create<mlir::tensor::ExtractSliceOp>(loc,
                                                        out_tensor_ty,
                                                        func.getArgument(0),
                                                        offsets,
                                                        sizes,
                                                        strides);
    b.create<mlir::func::ReturnOp>(loc, slice.getResult());
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
