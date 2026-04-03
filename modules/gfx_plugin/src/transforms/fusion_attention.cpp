// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transforms/fusion_patterns.hpp"
#include "transforms/fusion_utils.hpp"

#include "mlir/IR/Attributes.h"

namespace ov {
namespace gfx_plugin {
namespace {

using fusion_utils::has_single_forward_user;
using fusion_utils::is_constant_like_value;

bool is_scale_op(mlir::Operation* op, mlir::Value input, mlir::Value& scale_operand) {
    if (!op || op->getNumOperands() != 2 || op->getNumResults() != 1) {
        return false;
    }
    const auto name = op->getName().getStringRef();
    if (name == "gfx.Multiply") {
        if (op->getOperand(0) == input) {
            scale_operand = op->getOperand(1);
            return true;
        }
        if (op->getOperand(1) == input) {
            scale_operand = op->getOperand(0);
            return true;
        }
        return false;
    }
    if (name == "gfx.Divide") {
        if (op->getOperand(0) == input) {
            scale_operand = op->getOperand(1);
            return true;
        }
        return false;
    }
    return false;
}

bool match_prescaled_operand(mlir::Value value,
                             mlir::Operation* consumer,
                             mlir::Value& source_operand,
                             mlir::Operation*& scale_op,
                             mlir::Value& scale_operand) {
    source_operand = value;
    scale_op = nullptr;
    scale_operand = {};
    auto* def = value.getDefiningOp();
    if (!def || def->getNumResults() != 1) {
        return false;
    }
    const auto name = def->getName().getStringRef();
    if (name == "gfx.Multiply") {
        if (is_constant_like_value(def->getOperand(1)) && !is_constant_like_value(def->getOperand(0))) {
            scale_operand = def->getOperand(1);
            source_operand = def->getOperand(0);
        } else if (is_constant_like_value(def->getOperand(0)) && !is_constant_like_value(def->getOperand(1))) {
            scale_operand = def->getOperand(0);
            source_operand = def->getOperand(1);
        } else {
            return false;
        }
    } else if (name == "gfx.Divide") {
        if (!is_constant_like_value(def->getOperand(1)) || is_constant_like_value(def->getOperand(0))) {
            return false;
        }
        scale_operand = def->getOperand(1);
        source_operand = def->getOperand(0);
    } else {
        return false;
    }
    mlir::Operation* user = nullptr;
    if (!has_single_forward_user(def->getResult(0), user) || user != consumer) {
        return false;
    }
    scale_op = def;
    return source_operand != nullptr;
}

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

        mlir::Operation* next = nullptr;
        if (!has_single_forward_user(op->getResult(0), next)) {
            return mlir::failure();
        }

        mlir::Operation* pre_scale_lhs = nullptr;
        mlir::Operation* pre_scale_rhs = nullptr;
        mlir::Value lhs = op->getOperand(0);
        mlir::Value rhs = op->getOperand(1);
        mlir::Value lhs_scale_operand;
        mlir::Value rhs_scale_operand;
        (void)match_prescaled_operand(lhs, op, lhs, pre_scale_lhs, lhs_scale_operand);
        (void)match_prescaled_operand(rhs, op, rhs, pre_scale_rhs, rhs_scale_operand);

        mlir::Operation* scale = nullptr;
        mlir::Value score = op->getResult(0);
        mlir::Value scale_operand;
        if (is_scale_op(next, score, scale_operand)) {
            if (!is_constant_like_value(scale_operand)) {
                return mlir::failure();
            }
            scale = next;
            score = scale->getResult(0);
            if (!has_single_forward_user(score, next)) {
                return mlir::failure();
            }
        }

        mlir::Operation* softmax = next;
        if (softmax->getName().getStringRef() != "gfx.Softmax") {
            return mlir::failure();
        }
        if (softmax->getNumOperands() != 1 || softmax->getOperand(0) != score) {
            return mlir::failure();
        }

        mlir::Operation* matmul2 = nullptr;
        if (!has_single_forward_user(softmax->getResult(0), matmul2)) {
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
        auto lhs_scale_idx =
            pre_scale_lhs ? pre_scale_lhs->getAttrOfType<mlir::IntegerAttr>("gfx.node_index") : mlir::IntegerAttr{};
        auto rhs_scale_idx =
            pre_scale_rhs ? pre_scale_rhs->getAttrOfType<mlir::IntegerAttr>("gfx.node_index") : mlir::IntegerAttr{};
        auto scale_idx = scale ? scale->getAttrOfType<mlir::IntegerAttr>("gfx.node_index") : mlir::IntegerAttr{};
        auto sm_idx = softmax->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
        auto mm2_idx = matmul2->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
        if (!mm1_idx || !sm_idx || !mm2_idx || (scale && !scale_idx) ||
            (pre_scale_lhs && !lhs_scale_idx) || (pre_scale_rhs && !rhs_scale_idx)) {
            return mlir::failure();
        }

        mlir::OperationState state(matmul2->getLoc(), "gfx.FusedAttention");
        state.addOperands(lhs);
        state.addOperands(rhs);
        const bool has_scale = pre_scale_lhs || pre_scale_rhs || scale;
        if (pre_scale_lhs) {
            state.addOperands(lhs_scale_operand);
        }
        if (pre_scale_rhs) {
            state.addOperands(rhs_scale_operand);
        }
        if (scale) {
            state.addOperands(scale_operand);
        }
        if (other_input) {
            state.addOperands(other_input);
        }
        state.addTypes(matmul2->getResultTypes());
        state.addAttribute("gfx.node_index", mm1_idx);
        if (auto name = op->getAttrOfType<mlir::StringAttr>("gfx.node_name")) {
            state.addAttribute("gfx.node_name", name);
        }
        state.addAttribute("gfx.node_type",
                           rewriter.getStringAttr(has_scale ? "FusedAttentionScale" : "FusedAttention"));
        mlir::SmallVector<int64_t, 6> fused_nodes;
        llvm::SmallVector<int64_t, 2> prescale_nodes;
        if (pre_scale_lhs) {
            prescale_nodes.push_back(lhs_scale_idx.getInt());
        }
        if (pre_scale_rhs) {
            prescale_nodes.push_back(rhs_scale_idx.getInt());
        }
        llvm::sort(prescale_nodes);
        fused_nodes.append(prescale_nodes.begin(), prescale_nodes.end());
        fused_nodes.push_back(mm1_idx.getInt());
        if (scale) {
            fused_nodes.push_back(scale_idx.getInt());
        }
        fused_nodes.push_back(sm_idx.getInt());
        fused_nodes.push_back(mm2_idx.getInt());
        state.addAttribute("gfx.fused_nodes", rewriter.getI64ArrayAttr(fused_nodes));
        state.addAttribute("gfx.fusion_kind", rewriter.getStringAttr(has_scale ? "AttentionScale" : "Attention"));

        rewriter.setInsertionPoint(matmul2);
        auto* fused = rewriter.insert(mlir::Operation::create(state));
        rewriter.replaceOp(matmul2, fused->getResults());
        rewriter.eraseOp(softmax);
        if (scale) {
            rewriter.eraseOp(scale);
        }
        if (pre_scale_lhs) {
            rewriter.eraseOp(pre_scale_lhs);
        }
        if (pre_scale_rhs && pre_scale_rhs != pre_scale_lhs) {
            rewriter.eraseOp(pre_scale_rhs);
        }
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
