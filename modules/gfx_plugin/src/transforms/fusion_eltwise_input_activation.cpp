// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transforms/fusion_patterns.hpp"
#include "transforms/fusion_utils.hpp"

#include "mlir/IR/Attributes.h"

namespace ov {
namespace gfx_plugin {
namespace {

using fusion_utils::activation_alpha_or;
using fusion_utils::activation_kind_name;
using fusion_utils::has_single_user;

bool is_supported_input_activation(mlir::Operation* op, ActivationKind& kind) {
    if (!fusion_utils::is_supported_activation_op(op, kind)) {
        return false;
    }
    return kind == ActivationKind::Relu ||
           kind == ActivationKind::Sigmoid ||
           kind == ActivationKind::Tanh ||
           kind == ActivationKind::Gelu ||
           kind == ActivationKind::Swish ||
           kind == ActivationKind::HSwish ||
           kind == ActivationKind::HSigmoid;
}

struct EltwiseInputActivationFusionPattern final : public mlir::RewritePattern {
    EltwiseInputActivationFusionPattern(mlir::MLIRContext* ctx, FusionConfig config)
        : mlir::RewritePattern("gfx.Multiply", /*benefit=*/2, ctx), m_config(config) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation* op,
                                        mlir::PatternRewriter& rewriter) const override {
        if (!m_config.enable_fusion || op->getNumResults() != 1 || op->getNumOperands() != 2) {
            return mlir::failure();
        }

        mlir::Operation* activation = nullptr;
        size_t input_idx = 0;
        for (size_t i = 0; i < op->getNumOperands(); ++i) {
            auto* def = op->getOperand(i).getDefiningOp();
            ActivationKind kind = ActivationKind::Relu;
            if (def && is_supported_input_activation(def, kind)) {
                if (activation) {
                    return mlir::failure();
                }
                activation = def;
                input_idx = i;
            }
        }
        ActivationKind kind = ActivationKind::Relu;
        if (!activation || !is_supported_input_activation(activation, kind)) {
            return mlir::failure();
        }
        if (activation->getNumResults() != 1 || activation->getNumOperands() != 1) {
            return mlir::failure();
        }
        mlir::Operation* user = nullptr;
        if (!has_single_user(activation->getResult(0), user) || user != op) {
            return mlir::failure();
        }

        auto elt_idx = op->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
        auto act_idx = activation->getAttrOfType<mlir::IntegerAttr>("gfx.node_index");
        if (!elt_idx || !act_idx) {
            return mlir::failure();
        }

        llvm::SmallVector<mlir::Value, 2> operands;
        operands.reserve(op->getNumOperands());
        for (size_t i = 0; i < op->getNumOperands(); ++i) {
            operands.push_back(i == input_idx ? activation->getOperand(0) : op->getOperand(i));
        }

        mlir::OperationState state(op->getLoc(), "gfx.FusedEltwiseInputAct");
        state.addOperands(operands);
        state.addTypes(op->getResultTypes());
        state.addAttribute("gfx.node_index", elt_idx);
        if (auto name = op->getAttrOfType<mlir::StringAttr>("gfx.node_name")) {
            state.addAttribute("gfx.node_name", name);
        }
        state.addAttribute("gfx.node_type", rewriter.getStringAttr("FusedEltwiseInputAct"));
        state.addAttribute("gfx.fused_nodes",
                           rewriter.getI64ArrayAttr({elt_idx.getInt(), act_idx.getInt()}));
        state.addAttribute("gfx.fusion_kind", rewriter.getStringAttr("EltwiseInputActivation"));
        state.addAttribute("gfx.input_activation_kind",
                           rewriter.getStringAttr(activation_kind_name(kind)));
        state.addAttribute("gfx.input_activation_alpha",
                           rewriter.getF32FloatAttr(activation_alpha_or(activation, 0.0f)));
        state.addAttribute("gfx.input_activation_input",
                           rewriter.getI64IntegerAttr(static_cast<int64_t>(input_idx)));

        rewriter.setInsertionPoint(op);
        auto* fused = rewriter.insert(mlir::Operation::create(state));
        rewriter.replaceOp(op, fused->getResults());
        rewriter.eraseOp(activation);
        return mlir::success();
    }

private:
    FusionConfig m_config;
};

}  // namespace

void add_eltwise_input_activation_fusion_patterns(mlir::RewritePatternSet& patterns,
                                                  const FusionConfig& config) {
    if (!config.enable_fusion) {
        return;
    }
    patterns.add<EltwiseInputActivationFusionPattern>(patterns.getContext(), config);
}

}  // namespace gfx_plugin
}  // namespace ov
