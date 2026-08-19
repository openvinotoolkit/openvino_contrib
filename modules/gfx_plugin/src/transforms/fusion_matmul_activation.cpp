// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transforms/fusion_patterns.hpp"
#include "transforms/fusion_utils.hpp"

#include "mlir/IR/Attributes.h"

namespace ov {
namespace gfx_plugin {
namespace {

using fusion_utils::activation_kind_name;
using fusion_utils::activation_alpha_or;
using fusion_utils::has_single_user;
using fusion_utils::is_supported_activation_op;

struct MatMulActivationFusionPattern final : public mlir::RewritePattern {
    MatMulActivationFusionPattern(mlir::MLIRContext* ctx, FusionConfig config)
        : mlir::RewritePattern("gfx.MatMul", /*benefit=*/1, ctx), m_config(config) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation* op,
                                        mlir::PatternRewriter& rewriter) const override {
        if (!m_config.enable_fusion) {
            return mlir::failure();
        }
        if (op->getNumResults() != 1) {
            return mlir::failure();
        }

        mlir::Operation* user = nullptr;
        if (!has_single_user(op->getResult(0), user)) {
            return mlir::failure();
        }
        ActivationKind kind = ActivationKind::Relu;
        if (!is_supported_activation_op(user, kind)) {
            return mlir::failure();
        }
        if (user->getNumOperands() != 1 || user->getOperand(0) != op->getResult(0)) {
            return mlir::failure();
        }

        auto mm_idx = op->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
        auto act_idx = user->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
        if (!mm_idx || !act_idx) {
            return mlir::failure();
        }

        mlir::OperationState state(user->getLoc(), "gfx.FusedMatMulAct");
        state.addOperands(op->getOperands());
        state.addTypes(user->getResultTypes());
        state.addAttribute("gfx.node_index", mm_idx);
        if (auto name = op->getAttrOfType<mlir::StringAttr>("gfx.node_name")) {
            state.addAttribute("gfx.node_name", name);
        }
        state.addAttribute("gfx.node_type", rewriter.getStringAttr("FusedMatMulAct"));
        state.addAttribute("gfx.fused_nodes",
                           rewriter.getI64ArrayAttr({mm_idx.getInt(), act_idx.getInt()}));
        state.addAttribute("gfx.fusion_kind", rewriter.getStringAttr("MatMulActivation"));
        state.addAttribute("gfx.activation_kind",
                           rewriter.getStringAttr(activation_kind_name(kind)));
        const float alpha = activation_alpha_or(user, 0.0f);
        state.addAttribute("gfx.activation_alpha", rewriter.getF32FloatAttr(alpha));

        rewriter.setInsertionPoint(user);
        auto* fused = rewriter.insert(mlir::Operation::create(state));
        rewriter.replaceOp(user, fused->getResults());
        rewriter.eraseOp(op);
        return mlir::success();
    }

private:
    FusionConfig m_config;
};

}  // namespace

void add_matmul_activation_fusion_patterns(mlir::RewritePatternSet& patterns,
                                           const FusionConfig& config) {
    if (!config.enable_fusion) {
        return;
    }
    patterns.add<MatMulActivationFusionPattern>(patterns.getContext(), config);
}

}  // namespace gfx_plugin
}  // namespace ov
