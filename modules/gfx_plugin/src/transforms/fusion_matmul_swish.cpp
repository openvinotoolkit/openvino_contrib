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

bool is_matmul_op(mlir::Operation* op) {
    return op && op->getName().getStringRef() == "gfx.MatMul";
}

bool is_sigmoid_op(mlir::Operation* op) {
    return op && op->getName().getStringRef() == "gfx.Sigmoid";
}

bool is_mul_op(mlir::Operation* op) {
    return op && op->getName().getStringRef() == "gfx.Multiply";
}

struct MatMulSwishFusionPattern final : public mlir::RewritePattern {
    MatMulSwishFusionPattern(mlir::MLIRContext* ctx, FusionConfig config)
        : mlir::RewritePattern("gfx.MatMul", /*benefit=*/3, ctx), m_config(config) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation* op,
                                        mlir::PatternRewriter& rewriter) const override {
        if (!m_config.enable_fusion) {
            return mlir::failure();
        }
        if (!is_matmul_op(op) || op->getNumResults() != 1) {
            return mlir::failure();
        }

        auto mm_out = op->getResult(0);
        mlir::Operation* sigmoid = nullptr;
        mlir::Operation* mul = nullptr;
        for (auto* user : mm_out.getUsers()) {
            if (is_sigmoid_op(user)) {
                sigmoid = user;
            } else if (is_mul_op(user)) {
                mul = user;
            } else {
                return mlir::failure();
            }
        }
        if (!sigmoid || !mul) {
            return mlir::failure();
        }
        if (sigmoid->getNumOperands() != 1 || sigmoid->getOperand(0) != mm_out) {
            return mlir::failure();
        }
        if (mul->getNumOperands() != 2) {
            return mlir::failure();
        }
        auto sig_out = sigmoid->getResult(0);
        const bool mul_ok = (mul->getOperand(0) == mm_out && mul->getOperand(1) == sig_out) ||
                            (mul->getOperand(1) == mm_out && mul->getOperand(0) == sig_out);
        if (!mul_ok) {
            return mlir::failure();
        }
        mlir::Operation* sig_user = nullptr;
        if (!has_single_user(sig_out, sig_user) || sig_user != mul) {
            return mlir::failure();
        }

        auto mm_idx = op->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
        auto sig_idx = sigmoid->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
        auto mul_idx = mul->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
        if (!mm_idx || !sig_idx || !mul_idx) {
            return mlir::failure();
        }

        mlir::OperationState state(mul->getLoc(), "gfx.FusedMatMulAct");
        state.addOperands(op->getOperands());
        state.addTypes(mul->getResultTypes());
        state.addAttribute("gfx.node_index", mm_idx);
        if (auto name = op->getAttrOfType<mlir::StringAttr>("gfx.node_name")) {
            state.addAttribute("gfx.node_name", name);
        }
        state.addAttribute("gfx.node_type", rewriter.getStringAttr("FusedMatMulAct"));
        state.addAttribute("gfx.fused_nodes",
                           rewriter.getI64ArrayAttr({mm_idx.getInt(), sig_idx.getInt(), mul_idx.getInt()}));
        state.addAttribute("gfx.fusion_kind", rewriter.getStringAttr("MatMulActivation"));
        state.addAttribute("gfx.activation_kind", rewriter.getStringAttr("Swish"));
        state.addAttribute("gfx.activation_alpha", rewriter.getF32FloatAttr(0.0f));

        rewriter.setInsertionPoint(mul);
        auto* fused = rewriter.insert(mlir::Operation::create(state));
        rewriter.replaceOp(mul, fused->getResults());
        rewriter.eraseOp(sigmoid);
        rewriter.eraseOp(op);
        return mlir::success();
    }

private:
    FusionConfig m_config;
};

}  // namespace

void add_matmul_swish_fusion_patterns(mlir::RewritePatternSet& patterns,
                                      const FusionConfig& config) {
    if (!config.enable_fusion) {
        return;
    }
    patterns.add<MatMulSwishFusionPattern>(patterns.getContext(), config);
}

}  // namespace gfx_plugin
}  // namespace ov
