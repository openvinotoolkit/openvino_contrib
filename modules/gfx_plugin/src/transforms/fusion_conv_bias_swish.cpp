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

bool is_conv_op(mlir::Operation* op) {
    if (!op) {
        return false;
    }
    const auto name = op->getName().getStringRef();
    return name == "gfx.Convolution" || name == "gfx.GroupConvolution";
}

bool is_constant_op(mlir::Value value) {
    return fusion_utils::is_constant_like_value(value);
}

bool is_sigmoid_op(mlir::Operation* op) {
    return op && op->getName().getStringRef() == "gfx.Sigmoid";
}

bool is_mul_op(mlir::Operation* op) {
    return op && op->getName().getStringRef() == "gfx.Multiply";
}

struct ConvBiasSwishFusionPattern final : public mlir::RewritePattern {
    ConvBiasSwishFusionPattern(mlir::MLIRContext* ctx, FusionConfig config, llvm::StringRef conv_name)
        : mlir::RewritePattern(conv_name, /*benefit=*/3, ctx), m_config(config) {}

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
        if (add->getName().getStringRef() != "gfx.Add" || add->getNumOperands() != 2) {
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

        auto add_out = add->getResult(0);
        mlir::Operation* sigmoid = nullptr;
        mlir::Operation* mul = nullptr;
        for (auto* user : add_out.getUsers()) {
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
        if (sigmoid->getNumOperands() != 1 || sigmoid->getOperand(0) != add_out) {
            return mlir::failure();
        }
        if (mul->getNumOperands() != 2) {
            return mlir::failure();
        }
        auto sig_out = sigmoid->getResult(0);
        const bool mul_ok = (mul->getOperand(0) == add_out && mul->getOperand(1) == sig_out) ||
                            (mul->getOperand(1) == add_out && mul->getOperand(0) == sig_out);
        if (!mul_ok) {
            return mlir::failure();
        }
        mlir::Operation* sig_user = nullptr;
        if (!has_single_user(sig_out, sig_user) || sig_user != mul) {
            return mlir::failure();
        }

        auto conv_idx = op->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
        auto add_idx = add->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
        auto sig_idx = sigmoid->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
        auto mul_idx = mul->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
        if (!conv_idx || !add_idx || !sig_idx || !mul_idx) {
            return mlir::failure();
        }

        mlir::OperationState state(mul->getLoc(), "gfx.FusedConvBiasAct");
        state.addOperands(op->getOperands());
        state.addOperands(bias);
        state.addTypes(mul->getResultTypes());
        state.addAttribute("gfx.node_index", conv_idx);
        if (auto name = op->getAttrOfType<mlir::StringAttr>("gfx.node_name")) {
            state.addAttribute("gfx.node_name", name);
        }
        state.addAttribute("gfx.node_type", rewriter.getStringAttr("FusedConvBiasAct"));
        state.addAttribute(
            "gfx.fused_nodes",
            rewriter.getI64ArrayAttr({conv_idx.getInt(), add_idx.getInt(), sig_idx.getInt(), mul_idx.getInt()}));
        state.addAttribute("gfx.fusion_kind", rewriter.getStringAttr("ConvBiasActivation"));
        state.addAttribute("gfx.activation_kind", rewriter.getStringAttr("Swish"));
        state.addAttribute("gfx.activation_alpha", rewriter.getF32FloatAttr(0.0f));

        rewriter.setInsertionPoint(mul);
        auto* fused = rewriter.insert(mlir::Operation::create(state));
        rewriter.replaceOp(mul, fused->getResults());
        rewriter.eraseOp(sigmoid);
        rewriter.eraseOp(add);
        rewriter.eraseOp(op);
        return mlir::success();
    }

private:
    FusionConfig m_config;
};

}  // namespace

void add_conv_bias_swish_fusion_patterns(mlir::RewritePatternSet& patterns,
                                         const FusionConfig& config) {
    if (!config.enable_fusion) {
        return;
    }
    patterns.add<ConvBiasSwishFusionPattern>(patterns.getContext(), config, "gfx.Convolution");
    patterns.add<ConvBiasSwishFusionPattern>(patterns.getContext(), config, "gfx.GroupConvolution");
}

}  // namespace gfx_plugin
}  // namespace ov
