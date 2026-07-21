// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/gfx_mlir_type_utils.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/tile.hpp"

#include <optional>

namespace ov {
namespace gfx_plugin {

namespace {

std::optional<std::vector<int64_t>> try_get_const_i64(const std::shared_ptr<const ov::Node>& node) {
    auto c = ov::as_type_ptr<const ov::op::v0::Constant>(node);
    if (!c) {
        return std::nullopt;
    }
    return c->cast_vector<int64_t>();
}

}  // namespace

mlir::ModuleOp build_mlir_tile_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::memref::MemRefDialect, mlir::arith::ArithDialect,
                    mlir::scf::SCFDialect>();

    std::shared_ptr<const ov::op::v0::Tile> tile;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto t = ov::as_type_ptr<const ov::op::v0::Tile>(node)) {
            OPENVINO_ASSERT(!tile, "Tile MLIR builder: expected single Tile op");
            tile = t;
        }
    }
    OPENVINO_ASSERT(tile, "Tile MLIR builder: Tile op not found");

    const auto in_pshape = tile->get_input_partial_shape(0);
    const auto out_pshape = tile->get_output_partial_shape(0);
    OPENVINO_ASSERT(in_pshape.rank().is_static() && out_pshape.rank().is_static(),
                    "Tile MLIR: input/output ranks must be static");
    const size_t rank = static_cast<size_t>(in_pshape.rank().get_length());
    OPENVINO_ASSERT(rank == static_cast<size_t>(out_pshape.rank().get_length()),
                    "Tile MLIR: input/output rank mismatch");
    const auto repeats_pshape = tile->get_input_partial_shape(1);
    if (repeats_pshape.is_static()) {
        OPENVINO_ASSERT(ov::shape_size(repeats_pshape.to_shape()) == rank,
                        "Tile MLIR: repeats rank mismatch");
    }
    auto repeats = try_get_const_i64(tile->get_input_node_shared_ptr(1));
    if (repeats) {
        OPENVINO_ASSERT(repeats->size() == rank, "Tile MLIR: repeats rank mismatch");
    }
    if (in_pshape.is_static() && out_pshape.is_static()) {
        const auto in_shape = in_pshape.to_shape();
        const auto out_shape = out_pshape.to_shape();
        if (repeats) {
            for (size_t i = 0; i < rank; ++i) {
                const int64_t expected = static_cast<int64_t>(in_shape[i]) * (*repeats)[i];
                OPENVINO_ASSERT(expected == static_cast<int64_t>(out_shape[i]),
                                "Tile MLIR: output shape mismatch at axis ",
                                i);
            }
        }
    }

    const auto output_element_type = tile->get_output_element_type(0);
    auto elem_ty = to_mlir_type(output_element_type, ctx, /*fallback_f32=*/false,
                                /*allow_unsigned=*/true);
    const bool use_i64_lane_storage =
        output_element_type == ov::element::i64 ||
        output_element_type == ov::element::u64;

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    module->setAttr("gfx.prefer_parallel", mb.getBoolAttr(true));
    if (use_i64_lane_storage) {
        module->setAttr("gfx.i64_storage_i32_lanes", mb.getBoolAttr(true));
    }
    mb.setInsertionPointToStart(module.getBody());
    auto storage_ty = use_i64_lane_storage ? static_cast<mlir::Type>(mb.getI32Type()) : elem_ty;
    auto flat_data_ty = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, storage_ty);
    auto scalar_ty = mlir::MemRefType::get({1}, mb.getI32Type());
    auto shape_ty = mlir::MemRefType::get({static_cast<int64_t>(std::max<size_t>(rank, 1))},
                                          mb.getI32Type());
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "tile_main",
                                              mb.getFunctionType({flat_data_ty,
                                                                  flat_data_ty,
                                                                  scalar_ty,
                                                                  scalar_ty,
                                                                  shape_ty,
                                                                  shape_ty,
                                                                  shape_ty,
                                                                  shape_ty},
                                                                 {}));
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());

    auto c0 = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1 = b.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto c2 = b.create<mlir::arith::ConstantIndexOp>(loc, 2);
    auto load_i32_as_index = [&](mlir::OpBuilder& ib, mlir::Value buffer, mlir::Value index) -> mlir::Value {
        auto value = ib.create<mlir::memref::LoadOp>(loc, buffer, mlir::ValueRange{index}).getResult();
        return ib.create<mlir::arith::IndexCastOp>(loc, ib.getIndexType(), value).getResult();
    };
    auto c_out = load_i32_as_index(b, func.getArgument(2), c0);
    auto c_rank = load_i32_as_index(b, func.getArgument(3), c0);

    auto loop = b.create<mlir::scf::ParallelOp>(loc,
                                                mlir::ValueRange{c0},
                                                mlir::ValueRange{c_out},
                                                mlir::ValueRange{c1});
    {
        auto* body = loop.getBody();
        mlir::OpBuilder lb(body, body->begin());
        auto iv = loop.getInductionVars()[0];

        mlir::Value idx = iv;
        mlir::Value in_linear = lb.create<mlir::arith::ConstantIndexOp>(loc, 0);

        for (size_t d = 0; d < rank; ++d) {
            auto axis = lb.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(d));
            auto stride = load_i32_as_index(lb, func.getArgument(6), axis);
            auto out_i = lb.create<mlir::arith::DivUIOp>(loc, idx, stride).getResult();
            idx = lb.create<mlir::arith::RemUIOp>(loc, idx, stride).getResult();

            auto in_i = out_i;
            auto in_dim = load_i32_as_index(lb, func.getArgument(5), axis);
            in_i = lb.create<mlir::arith::RemUIOp>(loc, in_i, in_dim).getResult();

            auto in_stride = load_i32_as_index(lb, func.getArgument(7), axis);
            auto scaled = lb.create<mlir::arith::MulIOp>(loc, in_i, in_stride).getResult();
            in_linear = lb.create<mlir::arith::AddIOp>(loc, in_linear, scaled).getResult();
        }

        (void)c_rank;
        if (use_i64_lane_storage) {
            auto in_base = lb.create<mlir::arith::MulIOp>(loc, in_linear, c2).getResult();
            auto out_base = lb.create<mlir::arith::MulIOp>(loc, iv, c2).getResult();
            auto in_hi = lb.create<mlir::arith::AddIOp>(loc, in_base, c1).getResult();
            auto out_hi = lb.create<mlir::arith::AddIOp>(loc, out_base, c1).getResult();
            auto low = lb.create<mlir::memref::LoadOp>(loc, func.getArgument(0), mlir::ValueRange{in_base}).getResult();
            auto high = lb.create<mlir::memref::LoadOp>(loc, func.getArgument(0), mlir::ValueRange{in_hi}).getResult();
            lb.create<mlir::memref::StoreOp>(loc, low, func.getArgument(1), mlir::ValueRange{out_base});
            lb.create<mlir::memref::StoreOp>(loc, high, func.getArgument(1), mlir::ValueRange{out_hi});
        } else {
            auto val = lb.create<mlir::memref::LoadOp>(loc, func.getArgument(0), mlir::ValueRange{in_linear}).getResult();
            lb.create<mlir::memref::StoreOp>(loc, val, func.getArgument(1), mlir::ValueRange{iv});
        }
    }

    b.create<mlir::func::ReturnOp>(loc);
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
