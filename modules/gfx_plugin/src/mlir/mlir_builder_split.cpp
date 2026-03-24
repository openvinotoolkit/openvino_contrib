// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/gfx_mlir_type_utils.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/variadic_split.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
std::vector<size_t> extract_split_sizes(const std::shared_ptr<const ov::Node>& node,
                                        int64_t& axis_out,
                                        ov::Shape& input_shape,
                                        const ov::Shape* input_shape_override) {
    if (input_shape_override && !input_shape_override->empty()) {
        input_shape = *input_shape_override;
    }
    if (auto s = ov::as_type_ptr<const ov::op::v1::Split>(node)) {
        auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(s->input_value(1).get_node_shared_ptr());
        OPENVINO_ASSERT(axis_const, "Split axis must be constant");
        axis_out = axis_const->cast_vector<int64_t>().at(0);
        if (input_shape.empty()) {
            input_shape = s->get_input_shape(0);
        }
        const size_t parts = s->get_num_splits();
        const size_t axis_norm = axis_out >= 0 ? static_cast<size_t>(axis_out)
                                               : static_cast<size_t>(axis_out + static_cast<int64_t>(input_shape.size()));
        OPENVINO_ASSERT(axis_norm < input_shape.size(), "Split axis out of range");
        OPENVINO_ASSERT(input_shape.at(axis_norm) % parts == 0, "Split dimension not divisible by parts");
        size_t chunk = input_shape.at(axis_norm) / parts;
        return std::vector<size_t>(parts, chunk);
    } else if (auto vs = ov::as_type_ptr<const ov::op::v1::VariadicSplit>(node)) {
        auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(vs->input_value(1).get_node_shared_ptr());
        OPENVINO_ASSERT(axis_const, "VariadicSplit axis must be constant");
        axis_out = axis_const->cast_vector<int64_t>().at(0);
        if (input_shape.empty()) {
            input_shape = vs->get_input_shape(0);
        }
        auto lengths_const = ov::as_type_ptr<const ov::op::v0::Constant>(vs->input_value(2).get_node_shared_ptr());
        OPENVINO_ASSERT(lengths_const, "VariadicSplit lengths must be constant");
        auto lengths = lengths_const->cast_vector<int64_t>();
        std::vector<size_t> res;
        res.reserve(lengths.size());
        for (auto v : lengths) {
            OPENVINO_ASSERT(v >= 0, "VariadicSplit negative length not supported");
            res.push_back(static_cast<size_t>(v));
        }
        return res;
    }
    OPENVINO_THROW("Split MLIR: unsupported node type");
}

std::vector<int64_t> invert_permutation(const std::vector<int64_t>& permutation) {
    std::vector<int64_t> inverse(permutation.size(), -1);
    for (size_t i = 0; i < permutation.size(); ++i) {
        const auto axis = permutation[i];
        OPENVINO_ASSERT(axis >= 0 && axis < static_cast<int64_t>(permutation.size()),
                        "Split MLIR: permutation axis out of range");
        OPENVINO_ASSERT(inverse[static_cast<size_t>(axis)] < 0,
                        "Split MLIR: permutation axis repeated");
        inverse[static_cast<size_t>(axis)] = static_cast<int64_t>(i);
    }
    return inverse;
}

mlir::ArrayAttr make_i64_array_attr(mlir::OpBuilder& builder, const std::vector<int64_t>& values) {
    mlir::SmallVector<mlir::Attribute> attrs;
    attrs.reserve(values.size());
    for (auto value : values) {
        attrs.push_back(builder.getI64IntegerAttr(value));
    }
    return builder.getArrayAttr(attrs);
}
}  // namespace

mlir::ModuleOp build_mlir_split_from_model(const std::shared_ptr<const ov::Model>& model,
                                           mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::tensor::TensorDialect, mlir::arith::ArithDialect,
                    mlir::linalg::LinalgDialect>();

    std::shared_ptr<const ov::Node> split_node;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::as_type_ptr<const ov::op::v1::Split>(node) ||
            ov::as_type_ptr<const ov::op::v1::VariadicSplit>(node)) {
            OPENVINO_ASSERT(!split_node, "Split MLIR builder: expected single split node");
            split_node = node;
        }
    }
    OPENVINO_ASSERT(split_node, "Split MLIR builder: split node not found");

    int64_t axis = 0;
    ov::Shape input_shape;
    auto split_sizes = extract_split_sizes(split_node, axis, input_shape, nullptr);
    auto elem_ty = to_mlir_type(split_node->get_output_element_type(0),
                                ctx,
                                /*fallback_f32=*/false,
                                /*allow_unsigned=*/false,
                                /*allow_small_ints=*/true);

    mlir::SmallVector<int64_t> in_shape_vec;
    in_shape_vec.reserve(input_shape.size());
    for (auto v : input_shape) in_shape_vec.push_back(static_cast<int64_t>(v));
    auto in_tensor_ty = mlir::RankedTensorType::get(in_shape_vec, elem_ty);
    mlir::SmallVector<mlir::Type> out_types;
    out_types.reserve(split_sizes.size());
    int64_t axis_norm = axis;
    if (axis_norm < 0)
        axis_norm += static_cast<int64_t>(input_shape.size());
    for (size_t i = 0; i < split_sizes.size(); ++i) {
        ov::Shape out_shape = input_shape;
        out_shape[static_cast<size_t>(axis_norm)] = split_sizes[i];
        mlir::SmallVector<int64_t> out_shape_vec;
        out_shape_vec.reserve(out_shape.size());
        for (auto v : out_shape) out_shape_vec.push_back(static_cast<int64_t>(v));
        out_types.push_back(mlir::RankedTensorType::get(out_shape_vec, elem_ty));
    }

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({in_tensor_ty}, out_types);
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "split_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());

    // axis_norm computed above
    OPENVINO_ASSERT(axis_norm >= 0 && static_cast<size_t>(axis_norm) < input_shape.size(),
                    "Split MLIR: axis out of range");

    mlir::SmallVector<mlir::Value> results;
    results.reserve(split_sizes.size());
    int64_t axis_offset = 0;
    for (size_t i = 0; i < split_sizes.size(); ++i) {
        mlir::SmallVector<mlir::OpFoldResult> offsets;
        mlir::SmallVector<mlir::OpFoldResult> sizes;
        mlir::SmallVector<mlir::OpFoldResult> strides;
        offsets.reserve(input_shape.size());
        sizes.reserve(input_shape.size());
        strides.reserve(input_shape.size());
        for (size_t d = 0; d < input_shape.size(); ++d) {
            int64_t off = (d == static_cast<size_t>(axis_norm)) ? axis_offset : 0;
            int64_t sz = (d == static_cast<size_t>(axis_norm))
                             ? static_cast<int64_t>(split_sizes[i])
                             : static_cast<int64_t>(input_shape[d]);
            offsets.push_back(b.getIndexAttr(off));
            sizes.push_back(b.getIndexAttr(sz));
            strides.push_back(b.getIndexAttr(1));
        }
        auto slice = b.create<mlir::tensor::ExtractSliceOp>(loc,
                                                            func.getArgument(0),
                                                            offsets,
                                                            sizes,
                                                            strides);
        results.push_back(slice.getResult());
        axis_offset += static_cast<int64_t>(split_sizes[i]);
    }

    b.create<mlir::func::ReturnOp>(loc, results);
    return module;
}

mlir::ModuleOp build_mlir_split_from_node(const std::shared_ptr<const ov::Node>& node,
                                          mlir::MLIRContext& ctx,
                                          const ov::Shape& input_shape,
                                          const MlirInputTransformDesc* input_transform) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::tensor::TensorDialect, mlir::arith::ArithDialect,
                    mlir::linalg::LinalgDialect>();
    OPENVINO_ASSERT(node, "Split MLIR builder: node is null");

    int64_t axis = 0;
    ov::Shape logical_shape;
    auto split_sizes = extract_split_sizes(node, axis, logical_shape, nullptr);
    const ov::Shape source_shape =
        (input_transform && input_transform->has_transpose()) ? input_transform->source_shape : input_shape;
    const bool has_input_transform = input_transform && input_transform->has_transpose();
    if (!has_input_transform) {
        logical_shape = input_shape;
        split_sizes = extract_split_sizes(node, axis, logical_shape, &logical_shape);
    }
    auto elem_ty = to_mlir_type(node->get_output_element_type(0),
                                ctx,
                                /*fallback_f32=*/false,
                                /*allow_unsigned=*/false,
                                /*allow_small_ints=*/true);

    mlir::SmallVector<int64_t> in_shape_vec;
    in_shape_vec.reserve(source_shape.size());
    for (auto v : source_shape) in_shape_vec.push_back(static_cast<int64_t>(v));
    auto in_tensor_ty = mlir::RankedTensorType::get(in_shape_vec, elem_ty);

    mlir::SmallVector<mlir::Type> out_types;
    out_types.reserve(split_sizes.size());
    int64_t axis_norm = axis;
    if (axis_norm < 0)
        axis_norm += static_cast<int64_t>(logical_shape.size());
    for (size_t i = 0; i < split_sizes.size(); ++i) {
        ov::Shape out_shape = logical_shape;
        out_shape[static_cast<size_t>(axis_norm)] = split_sizes[i];
        mlir::SmallVector<int64_t> out_shape_vec;
        out_shape_vec.reserve(out_shape.size());
        for (auto v : out_shape) out_shape_vec.push_back(static_cast<int64_t>(v));
        out_types.push_back(mlir::RankedTensorType::get(out_shape_vec, elem_ty));
    }

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({in_tensor_ty}, out_types);
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "split_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());

    OPENVINO_ASSERT(axis_norm >= 0 && static_cast<size_t>(axis_norm) < logical_shape.size(),
                    "Split MLIR: axis out of range");

    mlir::SmallVector<mlir::Value> results;
    results.reserve(split_sizes.size());
    int64_t axis_offset = 0;
    std::vector<int64_t> inverse_permutation;
    if (has_input_transform) {
        OPENVINO_ASSERT(input_transform->transpose_permutation.size() == logical_shape.size(),
                        "Split MLIR: transform permutation rank mismatch");
        OPENVINO_ASSERT(source_shape.size() == logical_shape.size(),
                        "Split MLIR: source/logical rank mismatch");
        inverse_permutation = invert_permutation(input_transform->transpose_permutation);
        b.setInsertionPointToStart(module.getBody());
        module->setAttr("gfx.absorbed_input0_perm",
                        make_i64_array_attr(b, input_transform->transpose_permutation));
        b.setInsertionPointToStart(&func.getBody().front());
    }
    for (size_t i = 0; i < split_sizes.size(); ++i) {
        ov::Shape out_shape = logical_shape;
        out_shape[static_cast<size_t>(axis_norm)] = split_sizes[i];
        if (!has_input_transform) {
            mlir::SmallVector<mlir::OpFoldResult> offsets;
            mlir::SmallVector<mlir::OpFoldResult> sizes;
            mlir::SmallVector<mlir::OpFoldResult> strides;
            offsets.reserve(logical_shape.size());
            sizes.reserve(logical_shape.size());
            strides.reserve(logical_shape.size());
            for (size_t d = 0; d < logical_shape.size(); ++d) {
                int64_t off = (d == static_cast<size_t>(axis_norm)) ? axis_offset : 0;
                int64_t sz = (d == static_cast<size_t>(axis_norm))
                                 ? static_cast<int64_t>(split_sizes[i])
                                 : static_cast<int64_t>(logical_shape[d]);
                offsets.push_back(b.getIndexAttr(off));
                sizes.push_back(b.getIndexAttr(sz));
                strides.push_back(b.getIndexAttr(1));
            }
            auto slice = b.create<mlir::tensor::ExtractSliceOp>(loc,
                                                                func.getArgument(0),
                                                                offsets,
                                                                sizes,
                                                                strides);
            results.push_back(slice.getResult());
        } else {
            const auto source_axis = static_cast<size_t>(input_transform->transpose_permutation[static_cast<size_t>(axis_norm)]);
            OPENVINO_ASSERT(source_axis < source_shape.size(),
                            "Split MLIR: transformed source axis out of range");

            mlir::SmallVector<mlir::OpFoldResult> slice_offsets;
            mlir::SmallVector<mlir::OpFoldResult> slice_sizes;
            mlir::SmallVector<mlir::OpFoldResult> slice_strides;
            slice_offsets.reserve(source_shape.size());
            slice_sizes.reserve(source_shape.size());
            slice_strides.reserve(source_shape.size());
            mlir::SmallVector<int64_t> slice_shape_vec;
            slice_shape_vec.reserve(source_shape.size());
            for (size_t d = 0; d < source_shape.size(); ++d) {
                const int64_t off = (d == source_axis) ? axis_offset : 0;
                const int64_t sz = (d == source_axis) ? static_cast<int64_t>(split_sizes[i])
                                                      : static_cast<int64_t>(source_shape[d]);
                slice_offsets.push_back(b.getIndexAttr(off));
                slice_sizes.push_back(b.getIndexAttr(sz));
                slice_strides.push_back(b.getIndexAttr(1));
                slice_shape_vec.push_back(sz);
            }
            auto slice = b.create<mlir::tensor::ExtractSliceOp>(loc,
                                                                func.getArgument(0),
                                                                slice_offsets,
                                                                slice_sizes,
                                                                slice_strides);

            mlir::SmallVector<int64_t> out_shape_vec;
            out_shape_vec.reserve(out_shape.size());
            for (auto dim : out_shape) {
                out_shape_vec.push_back(static_cast<int64_t>(dim));
            }
            auto out_ty = mlir::RankedTensorType::get(out_shape_vec, elem_ty);
            auto empty = b.create<mlir::tensor::EmptyOp>(loc, out_shape_vec, elem_ty);

            mlir::SmallVector<mlir::AffineExpr> input_exprs;
            input_exprs.reserve(slice_shape_vec.size());
            for (size_t src_dim = 0; src_dim < slice_shape_vec.size(); ++src_dim) {
                const auto target_dim = static_cast<size_t>(inverse_permutation[src_dim]);
                input_exprs.push_back(mlir::getAffineDimExpr(static_cast<unsigned>(target_dim), &ctx));
            }
            mlir::SmallVector<mlir::AffineExpr> output_exprs;
            output_exprs.reserve(out_shape.size());
            for (size_t d = 0; d < out_shape.size(); ++d) {
                output_exprs.push_back(mlir::getAffineDimExpr(static_cast<unsigned>(d), &ctx));
            }
            auto input_map = mlir::AffineMap::get(static_cast<int64_t>(out_shape.size()),
                                                  0,
                                                  input_exprs,
                                                  &ctx);
            auto output_map = mlir::AffineMap::getMultiDimIdentityMap(static_cast<unsigned>(out_shape.size()), &ctx);
            llvm::SmallVector<mlir::utils::IteratorType> iterators(
                out_shape.size(),
                mlir::utils::IteratorType::parallel);

            auto generic = b.create<mlir::linalg::GenericOp>(
                loc,
                out_ty,
                mlir::ValueRange{slice.getResult()},
                mlir::ValueRange{empty.getResult()},
                mlir::ArrayRef<mlir::AffineMap>{input_map, output_map},
                mlir::ArrayRef<mlir::utils::IteratorType>(iterators));
            {
                auto& region = generic.getRegion();
                region.getBlocks().clear();
                auto* block = &region.emplaceBlock();
                block->addArguments({elem_ty, elem_ty}, {loc, loc});
                mlir::OpBuilder body(block, block->begin());
                body.create<mlir::linalg::YieldOp>(loc, block->getArgument(0));
            }
            results.push_back(generic.getResult(0));
        }
        axis_offset += static_cast<int64_t>(split_sizes[i]);
    }

    b.create<mlir::func::ReturnOp>(loc, results);
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
