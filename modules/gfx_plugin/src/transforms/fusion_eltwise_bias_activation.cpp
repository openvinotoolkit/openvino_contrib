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

bool is_supported_eltwise_op(mlir::Operation* op) {
    if (!op) {
        return false;
    }
    const auto name = op->getName().getStringRef();
    return name == "gfx.Add" || name == "gfx.Multiply" || name == "gfx.Maximum";
}

bool is_constant_op(mlir::Value value) {
    return fusion_utils::is_constant_like_value(value);
}

struct EltwiseBiasActivationFusionPattern final : public mlir::RewritePattern {
    EltwiseBiasActivationFusionPattern(mlir::MLIRContext* ctx, FusionConfig config, llvm::StringRef op_name)
        : mlir::RewritePattern(op_name, /*benefit=*/2, ctx), m_config(config) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation* op,
                                        mlir::PatternRewriter& rewriter) const override {
        if (!m_config.enable_fusion) {
            return mlir::failure();
        }
        if (!is_supported_eltwise_op(op) || op->getNumResults() != 1) {
            return mlir::failure();
        }

        mlir::Operation* add = nullptr;
        if (!has_single_user(op->getResult(0), add)) {
            return mlir::failure();
        }
        if (add->getName().getStringRef() != "gfx.Add") {
            return mlir::failure();
        }
        if (add->getNumOperands() != 2) {
            return mlir::failure();
        }

        mlir::Value bias;
        if (add->getOperand(0) == op->getResult(0)) {
            bias = add->getOperand(1);
        } else if (add->getOperand(1) == op->getResult(0)) {
            bias = add->getOperand(0);
        } else {
            return mlir::failure();
        }
        if (!is_constant_op(bias)) {
            return mlir::failure();
        }

        mlir::Operation* act = nullptr;
        if (!has_single_user(add->getResult(0), act)) {
            return mlir::failure();
        }
        ActivationKind kind = ActivationKind::Relu;
        if (!is_supported_activation_op(act, kind)) {
            return mlir::failure();
        }
        if (act->getNumOperands() != 1 || act->getOperand(0) != add->getResult(0)) {
            return mlir::failure();
        }

        auto elt_idx = op->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
        auto add_idx = add->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
        auto act_idx = act->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
        if (!elt_idx || !add_idx || !act_idx) {
            return mlir::failure();
        }

        mlir::OperationState state(act->getLoc(), "gfx.FusedEltwiseBiasAct");
        state.addOperands(op->getOperands());
        state.addOperands(bias);
        state.addTypes(act->getResultTypes());
        state.addAttribute("gfx.node_index", elt_idx);
        if (auto name = op->getAttrOfType<mlir::StringAttr>("gfx.node_name")) {
            state.addAttribute("gfx.node_name", name);
        }
        state.addAttribute("gfx.node_type", rewriter.getStringAttr("FusedEltwiseBiasAct"));
        state.addAttribute("gfx.fused_nodes",
                           rewriter.getI64ArrayAttr({elt_idx.getInt(), add_idx.getInt(), act_idx.getInt()}));
        state.addAttribute("gfx.fusion_kind", rewriter.getStringAttr("EltwiseBiasActivation"));
        state.addAttribute("gfx.activation_kind",
                           rewriter.getStringAttr(activation_kind_name(kind)));
        const float alpha = activation_alpha_or(act, 0.0f);
        state.addAttribute("gfx.activation_alpha", rewriter.getF32FloatAttr(alpha));

        rewriter.setInsertionPoint(act);
        auto* fused = rewriter.insert(mlir::Operation::create(state));
        rewriter.replaceOp(act, fused->getResults());
        rewriter.eraseOp(add);
        rewriter.eraseOp(op);
        return mlir::success();
    }

private:
    FusionConfig m_config;
};

}  // namespace

void add_eltwise_bias_activation_fusion_patterns(mlir::RewritePatternSet& patterns,
                                                 const FusionConfig& config) {
    if (!config.enable_fusion) {
        return;
    }
    patterns.add<EltwiseBiasActivationFusionPattern>(patterns.getContext(), config, "gfx.Add");
    patterns.add<EltwiseBiasActivationFusionPattern>(patterns.getContext(), config, "gfx.Multiply");
    patterns.add<EltwiseBiasActivationFusionPattern>(patterns.getContext(), config, "gfx.Maximum");
}

}  // namespace gfx_plugin
}  // namespace ov
