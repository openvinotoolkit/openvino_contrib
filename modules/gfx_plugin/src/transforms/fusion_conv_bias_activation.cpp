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

bool is_conv_op(mlir::Operation* op) {
    if (!op) {
        return false;
    }
    const auto name = op->getName().getStringRef();
    return name == "gfx.Convolution" || name == "gfx.GroupConvolution";
}

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

struct ConvBiasActivationFusionPattern final : public mlir::RewritePattern {
    ConvBiasActivationFusionPattern(mlir::MLIRContext* ctx, FusionConfig config, llvm::StringRef conv_name)
        : mlir::RewritePattern(conv_name, /*benefit=*/2, ctx), m_config(config) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation* op,
                                        mlir::PatternRewriter& rewriter) const override {
        if (!m_config.enable_fusion) {
            return mlir::failure();
        }
        if (!is_conv_op(op) || op->getNumResults() != 1) {
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

        auto conv_idx = op->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
        auto add_idx = add->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
        auto act_idx = act->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
        if (!conv_idx || !add_idx || !act_idx) {
            return mlir::failure();
        }

        mlir::OperationState state(act->getLoc(), "gfx.FusedConvBiasAct");
        state.addOperands(op->getOperands());
        state.addOperands(bias);
        state.addTypes(act->getResultTypes());
        state.addAttribute("gfx.node_index", conv_idx);
        if (auto name = op->getAttrOfType<mlir::StringAttr>("gfx.node_name")) {
            state.addAttribute("gfx.node_name", name);
        }
        state.addAttribute("gfx.node_type", rewriter.getStringAttr("FusedConvBiasAct"));
        state.addAttribute("gfx.fused_nodes",
                           rewriter.getI64ArrayAttr({conv_idx.getInt(), add_idx.getInt(), act_idx.getInt()}));
        state.addAttribute("gfx.fusion_kind", rewriter.getStringAttr("ConvBiasActivation"));
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

void add_conv_bias_activation_fusion_patterns(mlir::RewritePatternSet& patterns,
                                              const FusionConfig& config) {
    if (!config.enable_fusion) {
        return;
    }
    patterns.add<ConvBiasActivationFusionPattern>(patterns.getContext(), config, "gfx.Convolution");
    patterns.add<ConvBiasActivationFusionPattern>(patterns.getContext(), config, "gfx.GroupConvolution");
}

}  // namespace gfx_plugin
}  // namespace ov
