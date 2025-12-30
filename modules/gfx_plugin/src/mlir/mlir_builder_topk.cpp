// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/util/topk_base.hpp"

#include <algorithm>
#include <limits>

namespace ov {
namespace gfx_plugin {

namespace {
mlir::Type to_mlir_type(ov::element::Type et, mlir::MLIRContext& ctx) {
    switch (et) {
        case ov::element::f16: return mlir::Float16Type::get(&ctx);
        case ov::element::f32: return mlir::Float32Type::get(&ctx);
        case ov::element::i32: return mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Signed);
        case ov::element::i64: return mlir::IntegerType::get(&ctx, 64, mlir::IntegerType::Signed);
        case ov::element::u32: return mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Unsigned);
        case ov::element::u64: return mlir::IntegerType::get(&ctx, 64, mlir::IntegerType::Unsigned);
        default: OPENVINO_THROW("TopK MLIR: unsupported element type");
    }
}

mlir::Value make_init_value(mlir::OpBuilder& b,
                            mlir::Location loc,
                            mlir::Type elem_ty,
                            ov::op::TopKMode mode) {
    if (mlir::isa<mlir::FloatType>(elem_ty)) {
        double init = (mode == ov::op::TopKMode::MAX) ? -std::numeric_limits<double>::infinity()
                                                      : std::numeric_limits<double>::infinity();
        return b.create<mlir::arith::ConstantOp>(loc, mlir::FloatAttr::get(elem_ty, init)).getResult();
    }
    auto ity = mlir::cast<mlir::IntegerType>(elem_ty);
    int64_t init = 0;
    if (mode == ov::op::TopKMode::MAX) {
        init = ity.isUnsigned() ? 0 : std::numeric_limits<int64_t>::min();
    } else {
        init = ity.isUnsigned() ? -1 : std::numeric_limits<int64_t>::max();
    }
    return b.create<mlir::arith::ConstantOp>(loc, mlir::IntegerAttr::get(elem_ty, init)).getResult();
}
}  // namespace

mlir::ModuleOp build_mlir_topk_from_model(const std::shared_ptr<const ov::Model>& model,
                                          mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::tensor::TensorDialect, mlir::arith::ArithDialect,
                    mlir::scf::SCFDialect>();

    std::shared_ptr<const ov::op::util::TopKBase> topk;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto t = std::dynamic_pointer_cast<const ov::op::util::TopKBase>(node)) {
            OPENVINO_ASSERT(!topk, "TopK MLIR builder: expected single TopK op");
            topk = t;
        }
    }
    OPENVINO_ASSERT(topk, "TopK MLIR builder: TopK op not found");

    const auto in_shape = topk->get_input_shape(0);
    const auto out0_shape = topk->get_output_shape(0);
    const auto out1_shape = topk->get_output_shape(1);
    OPENVINO_ASSERT(!in_shape.empty(), "TopK MLIR: input shape must be static");
    OPENVINO_ASSERT(!out0_shape.empty() && !out1_shape.empty(), "TopK MLIR: output shapes must be static");

    const auto axis = static_cast<size_t>(topk->get_axis());
    const auto axis_dim = static_cast<int64_t>(in_shape.at(axis));
    const auto k = static_cast<int64_t>(out0_shape.at(axis));
    OPENVINO_ASSERT(k > 0, "TopK MLIR: k must be > 0");
    OPENVINO_ASSERT(k <= axis_dim, "TopK MLIR: k exceeds axis dimension");

    OPENVINO_ASSERT(topk->get_sort_type() == ov::op::TopKSortType::NONE ||
                        topk->get_sort_type() == ov::op::TopKSortType::SORT_VALUES,
                    "TopK MLIR: only NONE or SORT_VALUES supported");

    auto in_elem_ty = to_mlir_type(topk->get_input_element_type(0), ctx);
    auto out_val_ty = to_mlir_type(topk->get_output_element_type(0), ctx);
    auto out_idx_ty = to_mlir_type(topk->get_output_element_type(1), ctx);

    mlir::SmallVector<int64_t> in_dims(in_shape.begin(), in_shape.end());
    mlir::SmallVector<int64_t> out0_dims(out0_shape.begin(), out0_shape.end());
    mlir::SmallVector<int64_t> out1_dims(out1_shape.begin(), out1_shape.end());

    auto in_tensor_ty = mlir::RankedTensorType::get(in_dims, in_elem_ty);
    auto out0_tensor_ty = mlir::RankedTensorType::get(out0_dims, out_val_ty);
    auto out1_tensor_ty = mlir::RankedTensorType::get(out1_dims, out_idx_ty);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({in_tensor_ty}, {out0_tensor_ty, out1_tensor_ty});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "topk_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());

    auto out0_flat_ty = mlir::RankedTensorType::get({static_cast<int64_t>(ov::shape_size(out0_shape))}, out_val_ty);
    auto out1_flat_ty = mlir::RankedTensorType::get({static_cast<int64_t>(ov::shape_size(out1_shape))}, out_idx_ty);
    auto in_flat_ty = mlir::RankedTensorType::get({static_cast<int64_t>(ov::shape_size(in_shape))}, in_elem_ty);

    auto collapse = [&](size_t rank) {
        mlir::SmallVector<mlir::ReassociationIndices> reassoc;
        mlir::ReassociationIndices group;
        for (size_t i = 0; i < rank; ++i)
            group.push_back(static_cast<int64_t>(i));
        reassoc.push_back(group);
        return reassoc;
    };

    mlir::Value in_flat = func.getArgument(0);
    if (in_shape.size() > 1) {
        in_flat = b.create<mlir::tensor::CollapseShapeOp>(loc, in_flat_ty, in_flat, collapse(in_shape.size()));
    }

    mlir::Value out0_flat = b.create<mlir::tensor::EmptyOp>(
        loc, mlir::ArrayRef<int64_t>{static_cast<int64_t>(ov::shape_size(out0_shape))}, out_val_ty);
    mlir::Value out1_flat = b.create<mlir::tensor::EmptyOp>(
        loc, mlir::ArrayRef<int64_t>{static_cast<int64_t>(ov::shape_size(out1_shape))}, out_idx_ty);

    const int64_t inner = static_cast<int64_t>(
        ov::shape_size(ov::Shape(in_shape.begin() + axis + 1, in_shape.end())));
    const int64_t outer = static_cast<int64_t>(
        ov::shape_size(ov::Shape(in_shape.begin(), in_shape.begin() + axis)));
    const int64_t out_total = static_cast<int64_t>(ov::shape_size(out0_shape));

    auto c0 = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1 = b.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto c_outer = b.create<mlir::arith::ConstantIndexOp>(loc, outer);
    auto c_inner = b.create<mlir::arith::ConstantIndexOp>(loc, inner);
    auto c_k = b.create<mlir::arith::ConstantIndexOp>(loc, k);
    auto c_axis = b.create<mlir::arith::ConstantIndexOp>(loc, axis_dim);

    auto outer_loop = b.create<mlir::scf::ForOp>(loc, c0, c_outer, c1, mlir::ValueRange{out0_flat, out1_flat});
    {
        auto* body = outer_loop.getBody();
        mlir::OpBuilder ob(body, body->begin());
        auto outer_iv = outer_loop.getInductionVar();
        auto out0_acc = outer_loop.getRegionIterArgs()[0];
        auto out1_acc = outer_loop.getRegionIterArgs()[1];

        auto inner_loop = ob.create<mlir::scf::ForOp>(loc, c0, c_inner, c1, mlir::ValueRange{out0_acc, out1_acc});
        {
            auto* ibody = inner_loop.getBody();
            mlir::OpBuilder ib(ibody, ibody->begin());
            auto inner_iv = inner_loop.getInductionVar();
            auto out0_iacc = inner_loop.getRegionIterArgs()[0];
            auto out1_iacc = inner_loop.getRegionIterArgs()[1];

            auto k_loop = ib.create<mlir::scf::ForOp>(loc, c0, c_k, c1, mlir::ValueRange{out0_iacc, out1_iacc});
            {
                auto* kbody = k_loop.getBody();
                mlir::OpBuilder kb(kbody, kbody->begin());
                auto k_iv = k_loop.getInductionVar();
                auto out0_kacc = k_loop.getRegionIterArgs()[0];
                auto out1_kacc = k_loop.getRegionIterArgs()[1];

                auto best_val = make_init_value(kb, loc, out_val_ty, topk->get_mode());
                auto best_idx = kb.create<mlir::arith::ConstantIndexOp>(loc, 0).getResult();

                auto axis_loop = kb.create<mlir::scf::ForOp>(loc, c0, c_axis, c1, mlir::ValueRange{best_val, best_idx});
                {
                    auto* abody = axis_loop.getBody();
                    mlir::OpBuilder ab(abody, abody->begin());
                    auto a_iv = axis_loop.getInductionVar();
                    auto best_val_acc = axis_loop.getRegionIterArgs()[0];
                    auto best_idx_acc = axis_loop.getRegionIterArgs()[1];

                    auto selected_init = ab.create<mlir::arith::ConstantIntOp>(loc, 0, 1).getResult();
                    auto prev_loop = ab.create<mlir::scf::ForOp>(loc, c0, k_iv, c1, mlir::ValueRange{selected_init});
                    {
                        auto* pbody = prev_loop.getBody();
                        mlir::OpBuilder pb(pbody, pbody->begin());
                        auto p_iv = prev_loop.getInductionVar();
                        auto sel_acc = prev_loop.getRegionIterArgs()[0];

                        auto base = pb.create<mlir::arith::MulIOp>(loc, outer_iv, c_k).getResult();
                        base = pb.create<mlir::arith::AddIOp>(loc, base, p_iv).getResult();
                        base = pb.create<mlir::arith::MulIOp>(loc, base, c_inner).getResult();
                        auto out_idx = pb.create<mlir::arith::AddIOp>(loc, base, inner_iv).getResult();

                        auto stored = pb.create<mlir::tensor::ExtractOp>(loc, out1_kacc, mlir::ValueRange{out_idx}).getResult();
                        auto stored_idx = pb.create<mlir::arith::IndexCastOp>(loc, pb.getIndexType(), stored).getResult();
                        auto eq = pb.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, stored_idx, a_iv).getResult();
                        auto sel = pb.create<mlir::arith::OrIOp>(loc, sel_acc, eq).getResult();
                        pb.create<mlir::scf::YieldOp>(loc, sel);
                    }
                    auto selected = prev_loop.getResults()[0];

                    auto base = ab.create<mlir::arith::MulIOp>(loc, outer_iv, c_axis).getResult();
                    base = ab.create<mlir::arith::AddIOp>(loc, base, a_iv).getResult();
                    base = ab.create<mlir::arith::MulIOp>(loc, base, c_inner).getResult();
                    auto in_idx = ab.create<mlir::arith::AddIOp>(loc, base, inner_iv).getResult();
                    auto val = ab.create<mlir::tensor::ExtractOp>(loc, in_flat, mlir::ValueRange{in_idx}).getResult();

                    mlir::Value better;
                    if (mlir::isa<mlir::FloatType>(out_val_ty)) {
                        auto pred = topk->get_mode() == ov::op::TopKMode::MAX
                                        ? mlir::arith::CmpFPredicate::OGT
                                        : mlir::arith::CmpFPredicate::OLT;
                        better = ab.create<mlir::arith::CmpFOp>(loc, pred, val, best_val_acc).getResult();
                    } else {
                        auto ity = mlir::cast<mlir::IntegerType>(out_val_ty);
                        auto pred = topk->get_mode() == ov::op::TopKMode::MAX
                                        ? (ity.isUnsigned() ? mlir::arith::CmpIPredicate::ugt
                                                            : mlir::arith::CmpIPredicate::sgt)
                                        : (ity.isUnsigned() ? mlir::arith::CmpIPredicate::ult
                                                            : mlir::arith::CmpIPredicate::slt);
                        better = ab.create<mlir::arith::CmpIOp>(loc, pred, val, best_val_acc).getResult();
                    }

                    auto not_selected =
                        ab.create<mlir::arith::XOrIOp>(loc, selected,
                                                       ab.create<mlir::arith::ConstantIntOp>(loc, 1, 1))
                            .getResult();
                    auto take = ab.create<mlir::arith::AndIOp>(loc, better, not_selected).getResult();
                    auto new_best_val = ab.create<mlir::arith::SelectOp>(loc, take, val, best_val_acc).getResult();
                    auto new_best_idx = ab.create<mlir::arith::SelectOp>(loc, take, a_iv, best_idx_acc).getResult();
                    ab.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{new_best_val, new_best_idx});
                }

                auto best_val_final = axis_loop.getResults()[0];
                auto best_idx_final = axis_loop.getResults()[1];

                auto out_base = kb.create<mlir::arith::MulIOp>(loc, outer_iv, c_k).getResult();
                out_base = kb.create<mlir::arith::AddIOp>(loc, out_base, k_iv).getResult();
                out_base = kb.create<mlir::arith::MulIOp>(loc, out_base, c_inner).getResult();
                auto out_idx = kb.create<mlir::arith::AddIOp>(loc, out_base, inner_iv).getResult();

                auto idx_cast = kb.create<mlir::arith::IndexCastOp>(loc, out_idx_ty, best_idx_final).getResult();
                auto out0_updated = kb.create<mlir::tensor::InsertOp>(loc, best_val_final, out0_kacc,
                                                                      mlir::ValueRange{out_idx})
                                        .getResult();
                auto out1_updated = kb.create<mlir::tensor::InsertOp>(loc, idx_cast, out1_kacc,
                                                                      mlir::ValueRange{out_idx})
                                        .getResult();
                kb.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{out0_updated, out1_updated});
            }

            ib.create<mlir::scf::YieldOp>(loc, k_loop.getResults());
        }

        ob.create<mlir::scf::YieldOp>(loc, inner_loop.getResults());
    }

    out0_flat = outer_loop.getResults()[0];
    out1_flat = outer_loop.getResults()[1];

    mlir::Value out0_val = out0_flat;
    mlir::Value out1_val = out1_flat;
    if (out0_shape.size() > 1) {
        out0_val = b.create<mlir::tensor::ExpandShapeOp>(loc, out0_tensor_ty, out0_flat, collapse(out0_shape.size()));
    }
    if (out1_shape.size() > 1) {
        out1_val = b.create<mlir::tensor::ExpandShapeOp>(loc, out1_tensor_ty, out1_flat, collapse(out1_shape.size()));
    }

    b.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{out0_val, out1_val});
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
