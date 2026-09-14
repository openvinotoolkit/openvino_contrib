// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/gfx_mlir_type_utils.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/concat.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

mlir::Value add_index_values(mlir::OpBuilder& b, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) {
    return b.create<mlir::arith::AddIOp>(loc, lhs, rhs).getResult();
}

mlir::Value index_const(mlir::OpBuilder& b, mlir::Location loc, int64_t value) {
    return b.create<mlir::arith::ConstantIndexOp>(loc, value).getResult();
}

}  // namespace

mlir::ModuleOp build_mlir_concat_from_model(const std::shared_ptr<const ov::Model>& model,
                                            mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();

    std::shared_ptr<const ov::op::v0::Concat> concat;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto c = ov::as_type_ptr<const ov::op::v0::Concat>(node)) {
            OPENVINO_ASSERT(!concat, "Concat MLIR builder: expected single Concat");
            concat = c;
        }
    }
    OPENVINO_ASSERT(concat, "Concat MLIR builder: Concat op not found");

    const auto out_pshape = concat->get_output_partial_shape(0);
    OPENVINO_ASSERT(out_pshape.rank().is_static(), "Concat MLIR: output rank must be static");
    const size_t rank = static_cast<size_t>(out_pshape.rank().get_length());
    OPENVINO_ASSERT(rank > 0, "Concat MLIR: output rank must be non-zero");

    auto elem_ty = to_mlir_type(concat->get_output_element_type(0),
                                ctx,
                                /*fallback_f32=*/false,
                                /*allow_unsigned=*/false,
                                /*allow_small_ints=*/true);

    auto out_shape_vec = to_shape(out_pshape);
    auto out_tensor_ty = mlir::RankedTensorType::get(out_shape_vec, elem_ty);

    mlir::SmallVector<mlir::Type> input_types;
    input_types.reserve(concat->get_input_size());
    for (size_t i = 0; i < concat->get_input_size(); ++i) {
        auto in_shape = to_shape(concat->get_input_partial_shape(i));
        input_types.push_back(mlir::RankedTensorType::get(in_shape, elem_ty));
    }

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType(input_types, {out_tensor_ty});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "concat_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());

    int64_t axis = concat->get_axis();
    if (axis < 0) axis += static_cast<int64_t>(rank);
    OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < rank, "Concat MLIR: axis out of range");

    llvm::SmallVector<mlir::Value> out_dyn_dims;
    out_dyn_dims.reserve(rank);
    for (size_t dim = 0; dim < rank; ++dim) {
        if (out_shape_vec[dim] != mlir::ShapedType::kDynamic) {
            continue;
        }
        if (static_cast<int64_t>(dim) == axis) {
            mlir::Value total;
            int64_t static_total = 0;
            for (size_t i = 0; i < concat->get_input_size(); ++i) {
                const auto in_shape = to_shape(concat->get_input_partial_shape(i));
                OPENVINO_ASSERT(in_shape.size() == rank, "Concat MLIR: input rank mismatch");
                mlir::Value part;
                if (in_shape[dim] == mlir::ShapedType::kDynamic) {
                    part = b.create<mlir::tensor::DimOp>(loc,
                                                         func.getArgument(static_cast<unsigned>(i)),
                                                         static_cast<int64_t>(dim))
                               .getResult();
                } else {
                    static_total += in_shape[dim];
                    continue;
                }
                total = total ? add_index_values(b, loc, total, part) : part;
            }
            if (static_total != 0) {
                auto c = index_const(b, loc, static_total);
                total = total ? add_index_values(b, loc, total, c) : c;
            }
            OPENVINO_ASSERT(total, "Concat MLIR: dynamic concat axis must have runtime extent");
            out_dyn_dims.push_back(total);
        } else {
            mlir::Value dyn_dim;
            for (size_t i = 0; i < concat->get_input_size(); ++i) {
                const auto in_shape = to_shape(concat->get_input_partial_shape(i));
                OPENVINO_ASSERT(in_shape.size() == rank, "Concat MLIR: input rank mismatch");
                if (in_shape[dim] == mlir::ShapedType::kDynamic) {
                    dyn_dim = b.create<mlir::tensor::DimOp>(loc,
                                                            func.getArgument(static_cast<unsigned>(i)),
                                                            static_cast<int64_t>(dim))
                                  .getResult();
                    break;
                }
            }
            OPENVINO_ASSERT(dyn_dim, "Concat MLIR: dynamic non-concat dim must map to a runtime input dim");
            out_dyn_dims.push_back(dyn_dim);
        }
    }
    mlir::Value result = b.create<mlir::tensor::EmptyOp>(loc, out_shape_vec, elem_ty, out_dyn_dims);

    int64_t axis_offset_static = 0;
    mlir::Value axis_offset_dynamic;
    for (size_t i = 0; i < concat->get_input_size(); ++i) {
        auto in_shape = to_shape(concat->get_input_partial_shape(i));
        OPENVINO_ASSERT(in_shape.size() == rank, "Concat MLIR: input rank mismatch");
        mlir::Value src = func.getArgument(static_cast<unsigned>(i));
        mlir::SmallVector<mlir::OpFoldResult> offsets;
        mlir::SmallVector<mlir::OpFoldResult> sizes;
        mlir::SmallVector<mlir::OpFoldResult> strides;
        offsets.reserve(rank);
        sizes.reserve(rank);
        strides.reserve(rank);
        for (size_t dim = 0; dim < rank; ++dim) {
            if (static_cast<int64_t>(dim) == axis) {
                offsets.push_back(axis_offset_dynamic ? mlir::OpFoldResult(axis_offset_dynamic)
                                                      : mlir::OpFoldResult(b.getIndexAttr(axis_offset_static)));
            } else {
                offsets.push_back(b.getIndexAttr(0));
            }
            if (in_shape[dim] == mlir::ShapedType::kDynamic) {
                sizes.push_back(b.create<mlir::tensor::DimOp>(loc, src, static_cast<int64_t>(dim)).getResult());
            } else {
                sizes.push_back(b.getIndexAttr(in_shape[dim]));
            }
            strides.push_back(b.getIndexAttr(1));
        }
        result = b.create<mlir::tensor::InsertSliceOp>(loc, src, result, offsets, sizes, strides).getResult();
        const auto axis_dim = in_shape[static_cast<size_t>(axis)];
        if (axis_dim == mlir::ShapedType::kDynamic) {
            auto dim_value = b.create<mlir::tensor::DimOp>(loc, src, axis).getResult();
            if (axis_offset_dynamic) {
                axis_offset_dynamic = add_index_values(b, loc, axis_offset_dynamic, dim_value);
            } else if (axis_offset_static != 0) {
                axis_offset_dynamic = add_index_values(b, loc, index_const(b, loc, axis_offset_static), dim_value);
                axis_offset_static = 0;
            } else {
                axis_offset_dynamic = dim_value;
            }
        } else if (axis_offset_dynamic) {
            axis_offset_dynamic = add_index_values(b, loc, axis_offset_dynamic, index_const(b, loc, axis_dim));
        } else {
            axis_offset_static += axis_dim;
        }
    }

    b.create<mlir::func::ReturnOp>(loc, result);
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
