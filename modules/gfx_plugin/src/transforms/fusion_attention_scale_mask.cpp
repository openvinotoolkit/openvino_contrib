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

bool is_constant_op(mlir::Value value) {
    if (!value) {
        return false;
    }
    auto* def = value.getDefiningOp();
    if (!def) {
        return false;
    }
    return def->getName().getStringRef() == "gfx.Constant";
}

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

struct AttentionScaleMaskFusionPattern final : public mlir::RewritePattern {
    AttentionScaleMaskFusionPattern(mlir::MLIRContext* ctx, FusionConfig config)
        : mlir::RewritePattern("gfx.MatMul", /*benefit=*/5, ctx), m_config(config) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation* op,
                                        mlir::PatternRewriter& rewriter) const override {
        if (!m_config.enable_fusion) {
            return mlir::failure();
        }
        if (op->getNumResults() != 1) {
            return mlir::failure();
        }

        mlir::Operation* next = nullptr;
        if (!has_single_user(op->getResult(0), next)) {
            return mlir::failure();
        }

        mlir::Operation* scale = nullptr;
        mlir::Operation* add = nullptr;
        mlir::Operation* softmax = nullptr;
        mlir::Operation* matmul2 = nullptr;

        mlir::Value score = op->getResult(0);
        mlir::Value scale_operand;
        if (is_scale_op(next, score, scale_operand)) {
            if (!is_constant_op(scale_operand)) {
                return mlir::failure();
            }
            scale = next;
            score = scale->getResult(0);
            if (!has_single_user(score, next)) {
                return mlir::failure();
            }
        }

        mlir::Value add_input;
        if (next->getName().getStringRef() == "gfx.Add") {
            if (next->getNumOperands() != 2) {
                return mlir::failure();
            }
            if (next->getOperand(0) != score && next->getOperand(1) != score) {
                return mlir::failure();
            }
            add = next;
            add_input = score;
            score = add->getResult(0);
            if (!has_single_user(score, softmax)) {
                return mlir::failure();
            }
        } else {
            softmax = next;
        }

        if (!softmax || softmax->getName().getStringRef() != "gfx.Softmax") {
            return mlir::failure();
        }
        if (softmax->getNumOperands() != 1 || softmax->getOperand(0) != score) {
            return mlir::failure();
        }

        if (!has_single_user(softmax->getResult(0), matmul2)) {
            return mlir::failure();
        }
        if (matmul2->getName().getStringRef() != "gfx.MatMul" || matmul2->getNumOperands() != 2) {
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

        if (!scale && !add) {
            return mlir::failure();
        }

        auto mm1_idx = op->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
        auto sm_idx = softmax->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
        auto mm2_idx = matmul2->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
        if (!mm1_idx || !sm_idx || !mm2_idx) {
            return mlir::failure();
        }

        mlir::SmallVector<int64_t, 5> fused_nodes;
        fused_nodes.push_back(mm1_idx.getInt());
        if (scale) {
            auto scale_idx = scale->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
            if (!scale_idx) {
                return mlir::failure();
            }
            fused_nodes.push_back(scale_idx.getInt());
        }
        if (add) {
            auto add_idx = add->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
            if (!add_idx) {
                return mlir::failure();
            }
            fused_nodes.push_back(add_idx.getInt());
        }
        fused_nodes.push_back(sm_idx.getInt());
        fused_nodes.push_back(mm2_idx.getInt());

        mlir::OperationState state(matmul2->getLoc(), "gfx.FusedAttentionScaleMask");
        state.addOperands(op->getOperands());
        if (scale) {
            state.addOperands(scale_operand);
        }
        if (add) {
            mlir::Value mask = (add->getOperand(0) == add_input) ? add->getOperand(1) : add->getOperand(0);
            state.addOperands(mask);
        }
        if (other_input) {
            state.addOperands(other_input);
        }
        state.addTypes(matmul2->getResultTypes());
        state.addAttribute("gfx.node_index", mm1_idx);
        if (auto name = op->getAttrOfType<mlir::StringAttr>("gfx.node_name")) {
            state.addAttribute("gfx.node_name", name);
        }
        state.addAttribute("gfx.node_type", rewriter.getStringAttr("FusedAttentionScaleMask"));
        state.addAttribute("gfx.fused_nodes", rewriter.getI64ArrayAttr(fused_nodes));
        state.addAttribute("gfx.fusion_kind", rewriter.getStringAttr("AttentionScaleMask"));

        rewriter.setInsertionPoint(matmul2);
        auto* fused = rewriter.insert(mlir::Operation::create(state));
        rewriter.replaceOp(matmul2, fused->getResults());
        rewriter.eraseOp(softmax);
        if (add) {
            rewriter.eraseOp(add);
        }
        if (scale) {
            rewriter.eraseOp(scale);
        }
        rewriter.eraseOp(op);
        return mlir::success();
    }

private:
    FusionConfig m_config;
};

}  // namespace

void add_attention_scale_mask_fusion_patterns(mlir::RewritePatternSet& patterns,
                                              const FusionConfig& config) {
    if (!config.enable_fusion) {
        return;
    }
    patterns.add<AttentionScaleMaskFusionPattern>(patterns.getContext(), config);
}

}  // namespace gfx_plugin
}  // namespace ov
