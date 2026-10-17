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

struct ConvScaleFusionPattern final : public mlir::RewritePattern {
    ConvScaleFusionPattern(mlir::MLIRContext* ctx, FusionConfig config, llvm::StringRef conv_name)
        : mlir::RewritePattern(conv_name, /*benefit=*/1, ctx), m_config(config) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation* op,
                                        mlir::PatternRewriter& rewriter) const override {
        if (!m_config.enable_fusion) {
            return mlir::failure();
        }
        if (!is_conv_op(op) || op->getNumResults() != 1) {
            return mlir::failure();
        }

        mlir::Operation* mul = nullptr;
        if (!has_single_user(op->getResult(0), mul)) {
            return mlir::failure();
        }
        if (mul->getName().getStringRef() != "gfx.Multiply" || mul->getNumOperands() != 2) {
            return mlir::failure();
        }

        mlir::Value scale;
        if (mul->getOperand(0) == op->getResult(0)) {
            scale = mul->getOperand(1);
        } else if (mul->getOperand(1) == op->getResult(0)) {
            scale = mul->getOperand(0);
        } else {
            return mlir::failure();
        }
        if (!is_constant_op(scale)) {
            return mlir::failure();
        }

        auto conv_idx = op->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
        auto mul_idx = mul->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
        if (!conv_idx || !mul_idx) {
            return mlir::failure();
        }

        mlir::OperationState state(mul->getLoc(), "gfx.FusedConvScale");
        state.addOperands(op->getOperands());
        state.addOperands(scale);
        state.addTypes(mul->getResultTypes());
        state.addAttribute("gfx.node_index", conv_idx);
        if (auto name = op->getAttrOfType<mlir::StringAttr>("gfx.node_name")) {
            state.addAttribute("gfx.node_name", name);
        }
        state.addAttribute("gfx.node_type", rewriter.getStringAttr("FusedConvScale"));
        state.addAttribute("gfx.fused_nodes", rewriter.getI64ArrayAttr({conv_idx.getInt(), mul_idx.getInt()}));
        state.addAttribute("gfx.fusion_kind", rewriter.getStringAttr("ConvScale"));

        rewriter.setInsertionPoint(mul);
        auto* fused = rewriter.insert(mlir::Operation::create(state));
        rewriter.replaceOp(mul, fused->getResults());
        rewriter.eraseOp(op);
        return mlir::success();
    }

private:
    FusionConfig m_config;
};

}  // namespace

void add_conv_scale_fusion_patterns(mlir::RewritePatternSet& patterns,
                                    const FusionConfig& config) {
    if (!config.enable_fusion) {
        return;
    }
    patterns.add<ConvScaleFusionPattern>(patterns.getContext(), config, "gfx.Convolution");
    patterns.add<ConvScaleFusionPattern>(patterns.getContext(), config, "gfx.GroupConvolution");
}

}  // namespace gfx_plugin
}  // namespace ov
