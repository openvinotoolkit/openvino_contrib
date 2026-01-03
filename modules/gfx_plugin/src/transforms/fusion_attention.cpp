// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transforms/fusion_patterns.hpp"
#include "transforms/fusion_utils.hpp"

#include "mlir/IR/Attributes.h"

namespace ov {
namespace gfx_plugin {
namespace {

using fusion_utils::has_single_user;

struct AttentionFusionPattern final : public mlir::RewritePattern {
    AttentionFusionPattern(mlir::MLIRContext* ctx, FusionConfig config)
        : mlir::RewritePattern("gfx.MatMul", /*benefit=*/4, ctx), m_config(config) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation* op,
                                        mlir::PatternRewriter& rewriter) const override {
        if (!m_config.enable_fusion) {
            return mlir::failure();
        }
        if (op->getNumResults() != 1) {
            return mlir::failure();
        }

        mlir::Operation* softmax = nullptr;
        if (!has_single_user(op->getResult(0), softmax)) {
            return mlir::failure();
        }
        if (softmax->getName().getStringRef() != "gfx.Softmax") {
            return mlir::failure();
        }
        if (softmax->getNumOperands() != 1 || softmax->getOperand(0) != op->getResult(0)) {
            return mlir::failure();
        }

        mlir::Operation* matmul2 = nullptr;
        if (!has_single_user(softmax->getResult(0), matmul2)) {
            return mlir::failure();
        }
        if (matmul2->getName().getStringRef() != "gfx.MatMul") {
            return mlir::failure();
        }
        if (matmul2->getNumOperands() != 2) {
            return mlir::failure();
        }

        mlir::Value other_input;
        if (matmul2->getOperand(0) == softmax->getResult(0)) {
            other_input = matmul2->getOperand(1);
        } else if (matmul2->getOperand(1) == softmax->getResult(0)) {
            other_input = matmul2->getOperand(0);
        } else {
            return mlir::failure();
        }

        auto mm1_idx = op->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
        auto sm_idx = softmax->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
        auto mm2_idx = matmul2->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
        if (!mm1_idx || !sm_idx || !mm2_idx) {
            return mlir::failure();
        }

        mlir::OperationState state(matmul2->getLoc(), "gfx.FusedAttention");
        state.addOperands(op->getOperands());
        if (other_input) {
            state.addOperands(other_input);
        }
        state.addTypes(matmul2->getResultTypes());
        state.addAttribute("gfx.node_index", mm1_idx);
        if (auto name = op->getAttrOfType<mlir::StringAttr>("gfx.node_name")) {
            state.addAttribute("gfx.node_name", name);
        }
        state.addAttribute("gfx.node_type", rewriter.getStringAttr("FusedAttention"));
        state.addAttribute("gfx.fused_nodes",
                           rewriter.getI64ArrayAttr({mm1_idx.getInt(), sm_idx.getInt(), mm2_idx.getInt()}));
        state.addAttribute("gfx.fusion_kind", rewriter.getStringAttr("Attention"));

        rewriter.setInsertionPoint(matmul2);
        auto* fused = rewriter.insert(mlir::Operation::create(state));
        rewriter.replaceOp(matmul2, fused->getResults());
        rewriter.eraseOp(softmax);
        rewriter.eraseOp(op);
        return mlir::success();
    }

private:
    FusionConfig m_config;
};

}  // namespace

void add_attention_fusion_patterns(mlir::RewritePatternSet& patterns,
                                   const FusionConfig& config) {
    if (!config.enable_fusion) {
        return;
    }
    patterns.add<AttentionFusionPattern>(patterns.getContext(), config);
}

}  // namespace gfx_plugin
}  // namespace ov
