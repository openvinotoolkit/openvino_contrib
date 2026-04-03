// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/gfx_mlir_type_utils.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

void build_split_copy_result(mlir::OpBuilder& builder,
                             mlir::Location loc,
                             mlir::MLIRContext& ctx,
                             mlir::Value source,
                             mlir::Value output,
                             const ov::Shape& logical_out_shape,
                             size_t source_rank,
                             const std::function<mlir::Value(mlir::OpBuilder&,
                                                             llvm::ArrayRef<mlir::Value>,
                                                             size_t)>& make_input_index) {
    (void)ctx;
    auto c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    llvm::SmallVector<mlir::Value> out_dims;
    out_dims.reserve(logical_out_shape.size());
    for (auto dim : logical_out_shape) {
        out_dims.push_back(builder.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(dim)));
    }

    std::function<void(mlir::OpBuilder&, size_t, llvm::SmallVector<mlir::Value>&)> build_loops;
    build_loops = [&](mlir::OpBuilder& nested_builder,
                      size_t dim,
                      llvm::SmallVector<mlir::Value>& out_indices) {
        if (dim == logical_out_shape.size()) {
            llvm::SmallVector<mlir::Value> source_indices;
            source_indices.reserve(source_rank);
            for (size_t src_dim = 0; src_dim < source_rank; ++src_dim) {
                source_indices.push_back(make_input_index(nested_builder, out_indices, src_dim));
            }
            auto value = nested_builder.create<mlir::memref::LoadOp>(loc, source, source_indices).getResult();
            nested_builder.create<mlir::memref::StoreOp>(loc, value, output, out_indices);
            return;
        }

        auto loop = nested_builder.create<mlir::scf::ForOp>(loc, c0, out_dims[dim], c1);
        {
            mlir::OpBuilder::InsertionGuard guard(nested_builder);
            nested_builder.setInsertionPointToStart(loop.getBody());
            out_indices.push_back(loop.getInductionVar());
            build_loops(nested_builder, dim + 1, out_indices);
            out_indices.pop_back();
            nested_builder.create<mlir::scf::YieldOp>(loc);
        }
    };

    llvm::SmallVector<mlir::Value> out_indices;
    build_loops(builder, 0, out_indices);
}
}  // namespace

mlir::ModuleOp build_mlir_split_from_model(const std::shared_ptr<const ov::Model>& model,
                                           mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::tensor::TensorDialect, mlir::arith::ArithDialect,
                    mlir::linalg::LinalgDialect, mlir::scf::SCFDialect, mlir::memref::MemRefDialect>();

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
    auto in_memref_ty = mlir::MemRefType::get(in_shape_vec, elem_ty);
    mlir::SmallVector<mlir::Type> func_args;
    func_args.reserve(1 + split_sizes.size());
    func_args.push_back(in_memref_ty);
    int64_t axis_norm = axis;
    if (axis_norm < 0)
        axis_norm += static_cast<int64_t>(input_shape.size());
    for (size_t i = 0; i < split_sizes.size(); ++i) {
        ov::Shape out_shape = input_shape;
        out_shape[static_cast<size_t>(axis_norm)] = split_sizes[i];
        mlir::SmallVector<int64_t> out_shape_vec;
        out_shape_vec.reserve(out_shape.size());
        for (auto v : out_shape) out_shape_vec.push_back(static_cast<int64_t>(v));
        func_args.push_back(mlir::MemRefType::get(out_shape_vec, elem_ty));
    }

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType(func_args, {});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "split_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());

    // axis_norm computed above
    OPENVINO_ASSERT(axis_norm >= 0 && static_cast<size_t>(axis_norm) < input_shape.size(),
                    "Split MLIR: axis out of range");

    int64_t axis_offset = 0;
    for (size_t i = 0; i < split_sizes.size(); ++i) {
        ov::Shape out_shape = input_shape;
        out_shape[static_cast<size_t>(axis_norm)] = split_sizes[i];
        (void)build_split_copy_result(
            b,
            loc,
            ctx,
            func.getArgument(0),
            func.getArgument(i + 1),
            out_shape,
            input_shape.size(),
            [&](mlir::OpBuilder& nested_builder, llvm::ArrayRef<mlir::Value> out_indices, size_t src_dim) -> mlir::Value {
                if (src_dim != static_cast<size_t>(axis_norm) || axis_offset == 0) {
                    return out_indices[src_dim];
                }
                auto offset = nested_builder.create<mlir::arith::ConstantIndexOp>(loc, axis_offset);
                return nested_builder.create<mlir::arith::AddIOp>(loc, out_indices[src_dim], offset).getResult();
            });
        axis_offset += static_cast<int64_t>(split_sizes[i]);
    }

    b.create<mlir::func::ReturnOp>(loc);
    return module;
}

mlir::ModuleOp build_mlir_split_from_node(const std::shared_ptr<const ov::Node>& node,
                                          mlir::MLIRContext& ctx,
                                          const ov::Shape& input_shape,
                                          const MlirInputTransformDesc* input_transform) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::tensor::TensorDialect, mlir::arith::ArithDialect,
                    mlir::linalg::LinalgDialect, mlir::scf::SCFDialect, mlir::memref::MemRefDialect>();
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
    auto in_memref_ty = mlir::MemRefType::get(in_shape_vec, elem_ty);

    mlir::SmallVector<mlir::Type> func_args;
    func_args.reserve(1 + split_sizes.size());
    func_args.push_back(in_memref_ty);
    int64_t axis_norm = axis;
    if (axis_norm < 0)
        axis_norm += static_cast<int64_t>(logical_shape.size());
    for (size_t i = 0; i < split_sizes.size(); ++i) {
        ov::Shape out_shape = logical_shape;
        out_shape[static_cast<size_t>(axis_norm)] = split_sizes[i];
        mlir::SmallVector<int64_t> out_shape_vec;
        out_shape_vec.reserve(out_shape.size());
        for (auto v : out_shape) out_shape_vec.push_back(static_cast<int64_t>(v));
        func_args.push_back(mlir::MemRefType::get(out_shape_vec, elem_ty));
    }

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType(func_args, {});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "split_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());

    OPENVINO_ASSERT(axis_norm >= 0 && static_cast<size_t>(axis_norm) < logical_shape.size(),
                    "Split MLIR: axis out of range");

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
            (void)build_split_copy_result(
                b,
                loc,
                ctx,
                func.getArgument(0),
                func.getArgument(i + 1),
                out_shape,
                logical_shape.size(),
                [&](mlir::OpBuilder& nested_builder, llvm::ArrayRef<mlir::Value> out_indices, size_t src_dim) -> mlir::Value {
                    if (src_dim != static_cast<size_t>(axis_norm) || axis_offset == 0) {
                        return out_indices[src_dim];
                    }
                    auto offset = nested_builder.create<mlir::arith::ConstantIndexOp>(loc, axis_offset);
                    return nested_builder.create<mlir::arith::AddIOp>(loc, out_indices[src_dim], offset).getResult();
                });
        } else {
            const auto source_axis = static_cast<size_t>(input_transform->transpose_permutation[static_cast<size_t>(axis_norm)]);
            OPENVINO_ASSERT(source_axis < source_shape.size(),
                            "Split MLIR: transformed source axis out of range");
            (void)build_split_copy_result(
                b,
                loc,
                ctx,
                func.getArgument(0),
                func.getArgument(i + 1),
                out_shape,
                source_shape.size(),
                [&](mlir::OpBuilder& nested_builder, llvm::ArrayRef<mlir::Value> out_indices, size_t src_dim) -> mlir::Value {
                    const auto target_dim = static_cast<size_t>(inverse_permutation[src_dim]);
                    mlir::Value expr = out_indices[target_dim];
                    if (src_dim == source_axis && axis_offset != 0) {
                        auto offset = nested_builder.create<mlir::arith::ConstantIndexOp>(loc, axis_offset);
                        expr = nested_builder.create<mlir::arith::AddIOp>(loc, expr, offset).getResult();
                    }
                    return expr;
                });
        }
        axis_offset += static_cast<int64_t>(split_sizes[i]);
    }

    b.create<mlir::func::ReturnOp>(loc);
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
