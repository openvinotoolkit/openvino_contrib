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

struct ConvBatchNormActFusionPattern final : public mlir::RewritePattern {
    ConvBatchNormActFusionPattern(mlir::MLIRContext* ctx, FusionConfig config, llvm::StringRef conv_name)
        : mlir::RewritePattern(conv_name, /*benefit=*/3, ctx), m_config(config) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation* op,
                                        mlir::PatternRewriter& rewriter) const override {
        if (!m_config.enable_fusion) {
            return mlir::failure();
        }
        if (op->getNumResults() != 1) {
            return mlir::failure();
        }

        mlir::Operation* bn = nullptr;
        if (!has_single_user(op->getResult(0), bn)) {
            return mlir::failure();
        }
        if (bn->getName().getStringRef() != "gfx.BatchNormInference") {
            return mlir::failure();
        }
        if (bn->getNumOperands() != 5 || bn->getOperand(0) != op->getResult(0)) {
            return mlir::failure();
        }

        mlir::Operation* act = nullptr;
        if (!has_single_user(bn->getResult(0), act)) {
            return mlir::failure();
        }
        ActivationKind kind = ActivationKind::Relu;
        if (!is_supported_activation_op(act, kind)) {
            return mlir::failure();
        }
        if (act->getNumOperands() != 1 || act->getOperand(0) != bn->getResult(0)) {
            return mlir::failure();
        }

        auto conv_idx = op->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
        auto bn_idx = bn->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
        auto act_idx = act->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
        if (!conv_idx || !bn_idx || !act_idx) {
            return mlir::failure();
        }

        mlir::OperationState state(act->getLoc(), "gfx.FusedConvBnAct");
        state.addOperands(op->getOperands());
        state.addTypes(act->getResultTypes());
        state.addAttribute("gfx.node_index", conv_idx);
        if (auto name = op->getAttrOfType<mlir::StringAttr>("gfx.node_name")) {
            state.addAttribute("gfx.node_name", name);
        }
        state.addAttribute("gfx.node_type", rewriter.getStringAttr("FusedConvBnAct"));
        state.addAttribute("gfx.fused_nodes",
                           rewriter.getI64ArrayAttr({conv_idx.getInt(), bn_idx.getInt(), act_idx.getInt()}));
        state.addAttribute("gfx.fusion_kind", rewriter.getStringAttr("ConvBatchNormAct"));
        state.addAttribute("gfx.activation_kind",
                           rewriter.getStringAttr(activation_kind_name(kind)));
        const float alpha = activation_alpha_or(act, 0.0f);
        state.addAttribute("gfx.activation_alpha", rewriter.getF32FloatAttr(alpha));

        rewriter.setInsertionPoint(act);
        auto* fused = rewriter.insert(mlir::Operation::create(state));
        rewriter.replaceOp(act, fused->getResults());
        rewriter.eraseOp(bn);
        rewriter.eraseOp(op);
        return mlir::success();
    }

private:
    FusionConfig m_config;
};

}  // namespace

void add_conv_batchnorm_act_fusion_patterns(mlir::RewritePatternSet& patterns,
                                            const FusionConfig& config) {
    if (!config.enable_fusion) {
        return;
    }
    patterns.add<ConvBatchNormActFusionPattern>(patterns.getContext(), config, "gfx.Convolution");
    patterns.add<ConvBatchNormActFusionPattern>(patterns.getContext(), config, "gfx.GroupConvolution");
}

}  // namespace gfx_plugin
}  // namespace ov
