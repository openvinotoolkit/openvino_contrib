// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transforms/fusion_pass.hpp"

#include <unordered_map>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/result.hpp"
#include "transforms/fusion_patterns.hpp"
#include "llvm/Support/raw_ostream.h"

namespace ov {
namespace gfx_plugin {

namespace {

mlir::Type to_mlir_element_type(mlir::MLIRContext& ctx, const ov::element::Type& et) {
    switch (et) {
        case ov::element::f16:
            return mlir::Float16Type::get(&ctx);
        case ov::element::f32:
            return mlir::Float32Type::get(&ctx);
        case ov::element::i8:
            return mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::SignednessSemantics::Signed);
        case ov::element::u8:
            return mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::SignednessSemantics::Unsigned);
        case ov::element::i32:
            return mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::SignednessSemantics::Signed);
        case ov::element::i64:
            return mlir::IntegerType::get(&ctx, 64, mlir::IntegerType::SignednessSemantics::Signed);
        case ov::element::u32:
            return mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::SignednessSemantics::Unsigned);
        case ov::element::u64:
            return mlir::IntegerType::get(&ctx, 64, mlir::IntegerType::SignednessSemantics::Unsigned);
        default:
            return mlir::Float32Type::get(&ctx);
    }
}

mlir::Type to_mlir_tensor_type(mlir::MLIRContext& ctx,
                               const ov::PartialShape& shape,
                               const ov::element::Type& et) {
    auto elem_ty = to_mlir_element_type(ctx, et);
    if (!shape.rank().is_static()) {
        return mlir::UnrankedTensorType::get(elem_ty);
    }
    llvm::SmallVector<int64_t, 8> dims;
    dims.reserve(shape.rank().get_length());
    for (const auto& d : shape) {
        dims.push_back(d.is_dynamic() ? mlir::ShapedType::kDynamic
                                      : static_cast<int64_t>(d.get_length()));
    }
    return mlir::RankedTensorType::get(dims, elem_ty);
}

struct GraphBuilder {
    GraphBuilder(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx)
        : m_model(model), m_ctx(ctx), m_builder(&ctx) {}

    mlir::ModuleOp build() {
        OPENVINO_ASSERT(m_model, "Fusion graph builder: model is null");
        m_ctx.allowUnregisteredDialects();
        m_ctx.loadDialect<mlir::func::FuncDialect>();

        auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&m_ctx));
        const auto& params = m_model->get_parameters();
        const auto& results = m_model->get_results();

        llvm::SmallVector<mlir::Type, 8> input_types;
        input_types.reserve(params.size());
        for (const auto& param : params) {
            input_types.push_back(to_mlir_tensor_type(m_ctx,
                                                      param->get_output_partial_shape(0),
                                                      param->get_output_element_type(0)));
        }

        llvm::SmallVector<mlir::Type, 8> output_types;
        output_types.reserve(results.size());
        for (const auto& res : results) {
            auto src = res->input_value(0);
            output_types.push_back(to_mlir_tensor_type(m_ctx,
                                                       src.get_partial_shape(),
                                                       src.get_element_type()));
        }

        auto func_type = m_builder.getFunctionType(input_types, output_types);
        auto func = mlir::func::FuncOp::create(mlir::UnknownLoc::get(&m_ctx), "gfx_graph", func_type);
        func.addEntryBlock();
        module.push_back(func);

        auto& block = func.getBody().front();
        m_builder.setInsertionPointToStart(&block);

        std::unordered_map<const ov::Node*, llvm::SmallVector<mlir::Value, 4>> value_map;
        value_map.reserve(m_model->get_ordered_ops().size());

        for (size_t i = 0; i < params.size(); ++i) {
            value_map[params[i].get()].push_back(func.getArgument(static_cast<unsigned>(i)));
        }

        const auto ordered_ops = m_model->get_ordered_ops();
        for (size_t idx = 0; idx < ordered_ops.size(); ++idx) {
            const auto& node = ordered_ops[idx];
            if (ov::as_type_ptr<ov::op::v0::Parameter>(node)) {
                continue;
            }
            if (ov::as_type_ptr<ov::op::v0::Result>(node)) {
                continue;
            }

            const std::string type_name = node->get_type_name();
            const std::string op_name = std::string("gfx.") + type_name;
            mlir::OperationState state(mlir::UnknownLoc::get(&m_ctx), op_name);

            llvm::SmallVector<mlir::Value, 8> operands;
            operands.reserve(node->get_input_size());
            for (size_t i = 0; i < node->get_input_size(); ++i) {
                auto input = node->input_value(i);
                const auto* src = input.get_node();
                auto it = value_map.find(src);
                OPENVINO_ASSERT(it != value_map.end(), "Fusion graph builder: missing input for node ",
                                node->get_friendly_name());
                const size_t out_idx = input.get_index();
                OPENVINO_ASSERT(out_idx < it->second.size(),
                                "Fusion graph builder: bad output index for node ",
                                node->get_friendly_name());
                operands.push_back(it->second[out_idx]);
            }

            llvm::SmallVector<mlir::Type, 4> result_types;
            result_types.reserve(node->get_output_size());
            for (size_t o = 0; o < node->get_output_size(); ++o) {
                result_types.push_back(to_mlir_tensor_type(m_ctx,
                                                           node->get_output_partial_shape(o),
                                                           node->get_output_element_type(o)));
            }

            if (ov::as_type_ptr<ov::op::v0::Constant>(node)) {
                state.operands.clear();
            } else {
                state.addOperands(operands);
            }
            state.addTypes(result_types);
            state.addAttribute("gfx.node_index", m_builder.getI64IntegerAttr(static_cast<int64_t>(idx)));
            state.addAttribute("gfx.node_name", m_builder.getStringAttr(node->get_friendly_name()));
            state.addAttribute("gfx.node_type", m_builder.getStringAttr(type_name));

            if (auto elu = ov::as_type_ptr<const ov::op::v0::Elu>(node)) {
                state.addAttribute("gfx.activation_alpha",
                                   m_builder.getF32FloatAttr(static_cast<float>(elu->get_alpha())));
            }
            if (auto prelu = ov::as_type_ptr<const ov::op::v0::PRelu>(node)) {
                auto slope_node = prelu->get_input_node_shared_ptr(1);
                auto slope_const = std::dynamic_pointer_cast<const ov::op::v0::Constant>(slope_node);
                if (slope_const && ov::shape_size(slope_const->get_shape()) == 1) {
                    const auto vals = slope_const->cast_vector<float>();
                    if (!vals.empty()) {
                        state.addAttribute("gfx.activation_alpha",
                                           m_builder.getF32FloatAttr(vals.front()));
                    }
                }
            }

            auto* op = mlir::Operation::create(state);
            m_builder.insert(op);
            llvm::SmallVector<mlir::Value, 4> outputs;
            outputs.reserve(op->getNumResults());
            for (auto res : op->getResults()) {
                outputs.push_back(res);
            }
            value_map[node.get()] = std::move(outputs);
        }

        llvm::SmallVector<mlir::Value, 8> ret_vals;
        ret_vals.reserve(results.size());
        for (const auto& res : results) {
            auto input = res->input_value(0);
            const auto* src = input.get_node();
            auto it = value_map.find(src);
            OPENVINO_ASSERT(it != value_map.end(), "Fusion graph builder: missing result source");
            const size_t out_idx = input.get_index();
            OPENVINO_ASSERT(out_idx < it->second.size(), "Fusion graph builder: invalid result output index");
            ret_vals.push_back(it->second[out_idx]);
        }
        m_builder.create<mlir::func::ReturnOp>(mlir::UnknownLoc::get(&m_ctx), ret_vals);
        return module;
    }

private:
    std::shared_ptr<const ov::Model> m_model;
    mlir::MLIRContext& m_ctx;
    mlir::OpBuilder m_builder;
};

std::optional<ActivationKind> parse_activation_kind(llvm::StringRef name) {
    if (name == "Relu") {
        return ActivationKind::Relu;
    }
    if (name == "Sigmoid") {
        return ActivationKind::Sigmoid;
    }
    if (name == "Tanh") {
        return ActivationKind::Tanh;
    }
    if (name == "Elu") {
        return ActivationKind::Elu;
    }
    if (name == "Prelu") {
        return ActivationKind::Prelu;
    }
    if (name == "Gelu") {
        return ActivationKind::Gelu;
    }
    if (name == "Swish") {
        return ActivationKind::Swish;
    }
    if (name == "HSwish") {
        return ActivationKind::HSwish;
    }
    if (name == "HSigmoid") {
        return ActivationKind::HSigmoid;
    }
    if (name == "Abs") {
        return ActivationKind::Abs;
    }
    if (name == "Sign") {
        return ActivationKind::Sign;
    }
    return std::nullopt;
}

void run_fusion_passes(mlir::ModuleOp module, const FusionConfig& config) {
    if (!config.enable_fusion) {
        return;
    }
    mlir::RewritePatternSet patterns(module.getContext());
    add_attention_scale_mask_fusion_patterns(patterns, config);
    add_attention_fusion_patterns(patterns, config);
    add_conv_batchnorm_act_fusion_patterns(patterns, config);
    add_conv_batchnorm_fusion_patterns(patterns, config);
    add_conv_bias_activation_fusion_patterns(patterns, config);
    add_conv_bias_fusion_patterns(patterns, config);
    add_conv_activation_fusion_patterns(patterns, config);
    add_eltwise_activation_fusion_patterns(patterns, config);
    add_eltwise_bias_activation_fusion_patterns(patterns, config);
    add_eltwise_bias_fusion_patterns(patterns, config);
    add_matmul_activation_fusion_patterns(patterns, config);
    add_matmul_bias_activation_fusion_patterns(patterns, config);
    add_matmul_bias_fusion_patterns(patterns, config);
    mlir::GreedyRewriteConfig cfg;
    cfg.setMaxIterations(2);
    if (mlir::failed(mlir::applyPatternsGreedily(module, std::move(patterns), cfg))) {
        OPENVINO_THROW("GFX fusion pass failed");
    }
}

FusionPlan extract_plan(mlir::ModuleOp module) {
    FusionPlan plan;
    auto func = module.lookupSymbol<mlir::func::FuncOp>("gfx_graph");
    if (!func) {
        return plan;
    }
    auto& block = func.getBody().front();
    for (auto& op : block) {
        if (llvm::isa<mlir::func::ReturnOp>(&op)) {
            continue;
        }
        const auto name = op.getName().getStringRef();
        if (name == "gfx.Constant") {
            continue;
        }
        auto fused_nodes = op.getAttrOfType<mlir::ArrayAttr>("gfx.fused_nodes");
        if (!fused_nodes) {
            continue;
        }
        FusionGroup group;
        for (auto attr : fused_nodes) {
            if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
                group.node_indices.push_back(static_cast<size_t>(int_attr.getInt()));
            }
        }
        if (auto kind_attr = op.getAttrOfType<mlir::StringAttr>("gfx.fusion_kind")) {
            group.kind = kind_attr.str();
        }
        if (auto act_attr = op.getAttrOfType<mlir::StringAttr>("gfx.activation_kind")) {
            group.activation = parse_activation_kind(act_attr.getValue());
        }
        if (auto alpha_attr = op.getAttrOfType<mlir::FloatAttr>("gfx.activation_alpha")) {
            group.activation_alpha = static_cast<float>(alpha_attr.getValueAsDouble());
        }
        if (!group.node_indices.empty()) {
            plan.groups.emplace_back(std::move(group));
        }
    }
    return plan;
}

}  // namespace

FusionPlan build_fusion_plan(const std::shared_ptr<const ov::Model>& model,
                             const FusionConfig& config) {
    FusionPlan plan;
    if (!model) {
        return plan;
    }
    if (!config.enable_fusion) {
        return plan;
    }
    mlir::MLIRContext ctx;
    GraphBuilder builder(model, ctx);
    auto module = builder.build();

    if (config.debug_dump_ir) {
        llvm::errs() << "[GFX][Fusion] MLIR graph before fusion:\n";
        module.print(llvm::errs());
        llvm::errs() << "\n";
    }

    run_fusion_passes(module, config);

    if (config.debug_dump_ir) {
        llvm::errs() << "[GFX][Fusion] MLIR graph after fusion:\n";
        module.print(llvm::errs());
        llvm::errs() << "\n";
    }

    plan = extract_plan(module);
    return plan;
}

}  // namespace gfx_plugin
}  // namespace ov
