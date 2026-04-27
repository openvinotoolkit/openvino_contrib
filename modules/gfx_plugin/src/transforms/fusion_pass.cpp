// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transforms/fusion_pass.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/mlir_support.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/batch_norm.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/op/transpose.hpp"
#include "transforms/fusion_patterns.hpp"
#include "llvm/Support/raw_ostream.h"

namespace ov {
namespace gfx_plugin {

namespace {

bool read_const_f32(const std::shared_ptr<const ov::op::v0::Constant>& constant,
                    std::vector<float>& out) {
    if (!constant) {
        return false;
    }
    const auto et = constant->get_element_type();
    const size_t count = shape_size(constant->get_shape());
    out.resize(count);
    if (count == 0) {
        return false;
    }
    if (et == ov::element::f32) {
        const float* src = constant->get_data_ptr<float>();
        std::copy(src, src + count, out.begin());
        return true;
    }
    if (et == ov::element::f16) {
        const ov::float16* src = constant->get_data_ptr<ov::float16>();
        for (size_t i = 0; i < count; ++i) {
            out[i] = static_cast<float>(src[i]);
        }
        return true;
    }
    return false;
}

bool extract_bias_params(const std::shared_ptr<const ov::Node>& node, BiasParams& out) {
    auto add = ov::as_type_ptr<const ov::op::v1::Add>(node);
    if (!add) {
        return false;
    }

    std::shared_ptr<const ov::op::v0::Constant> bias_const;
    if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(add->get_input_node_shared_ptr(0))) {
        bias_const = c;
    } else if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(add->get_input_node_shared_ptr(1))) {
        bias_const = c;
    } else {
        return false;
    }

    BiasParams params{};
    if (!read_const_f32(bias_const, params.values)) {
        return false;
    }
    params.element_type = bias_const->get_element_type();
    params.shape.clear();
    const auto& shape = bias_const->get_shape();
    params.shape.reserve(shape.size());
    for (auto dim : shape) {
        params.shape.push_back(static_cast<int64_t>(dim));
    }
    if (params.values.empty()) {
        return false;
    }
    if (shape_size(shape) != params.values.size()) {
        return false;
    }
    out = std::move(params);
    return true;
}

bool extract_batchnorm_params(const std::shared_ptr<const ov::Node>& node, BatchNormParams& out) {
    auto bn = ov::as_type_ptr<const ov::op::v5::BatchNormInference>(node);
    if (!bn) {
        return false;
    }

    auto gamma = std::dynamic_pointer_cast<const ov::op::v0::Constant>(bn->get_input_node_shared_ptr(1));
    auto beta = std::dynamic_pointer_cast<const ov::op::v0::Constant>(bn->get_input_node_shared_ptr(2));
    auto mean = std::dynamic_pointer_cast<const ov::op::v0::Constant>(bn->get_input_node_shared_ptr(3));
    auto var = std::dynamic_pointer_cast<const ov::op::v0::Constant>(bn->get_input_node_shared_ptr(4));
    if (!gamma || !beta || !mean || !var) {
        return false;
    }

    BatchNormParams params{};
    if (!read_const_f32(gamma, params.gamma) ||
        !read_const_f32(beta, params.beta) ||
        !read_const_f32(mean, params.mean) ||
        !read_const_f32(var, params.var)) {
        return false;
    }

    const size_t channels = params.gamma.size();
    if (channels == 0 || params.beta.size() != channels || params.mean.size() != channels ||
        params.var.size() != channels) {
        return false;
    }

    if (bn->get_input_partial_shape(0).rank().is_static()) {
        const auto& in_shape = bn->get_input_partial_shape(0);
        if (in_shape.rank().get_length() >= 2 && in_shape[1].is_static()) {
            const size_t expected = static_cast<size_t>(in_shape[1].get_length());
            if (expected != channels) {
                return false;
            }
        }
    }

    params.epsilon = static_cast<float>(bn->get_eps_value());
    out = std::move(params);
    return true;
}

bool extract_scale_as_batchnorm_params(const std::shared_ptr<const ov::Node>& scale_node,
                                       const std::shared_ptr<const ov::Node>& producer_node,
                                       BatchNormParams& out) {
    auto mul = ov::as_type_ptr<const ov::op::v1::Multiply>(scale_node);
    if (!mul || !producer_node) {
        return false;
    }

    std::shared_ptr<const ov::op::v0::Constant> scale_const;
    if (mul->get_input_node_shared_ptr(0) == producer_node) {
        scale_const = std::dynamic_pointer_cast<const ov::op::v0::Constant>(mul->get_input_node_shared_ptr(1));
    } else if (mul->get_input_node_shared_ptr(1) == producer_node) {
        scale_const = std::dynamic_pointer_cast<const ov::op::v0::Constant>(mul->get_input_node_shared_ptr(0));
    } else {
        return false;
    }
    if (!scale_const) {
        return false;
    }

    std::vector<float> raw_values;
    if (!read_const_f32(scale_const, raw_values) || raw_values.empty()) {
        return false;
    }

    const auto& out_pshape = producer_node->get_output_partial_shape(0);
    if (!out_pshape.rank().is_static() || out_pshape.rank().get_length() < 2 || !out_pshape[1].is_static()) {
        return false;
    }
    const size_t channels = static_cast<size_t>(out_pshape[1].get_length());
    if (channels == 0) {
        return false;
    }

    std::vector<float> gamma(channels, 1.0f);
    const auto scale_shape = scale_const->get_shape();
    if (raw_values.size() == 1) {
        std::fill(gamma.begin(), gamma.end(), raw_values.front());
    } else {
        const size_t out_rank = static_cast<size_t>(out_pshape.rank().get_length());
        if (scale_shape.size() > out_rank) {
            return false;
        }
        std::vector<size_t> aligned_shape(out_rank, 1);
        const size_t offset = out_rank - scale_shape.size();
        for (size_t i = 0; i < scale_shape.size(); ++i) {
            aligned_shape[offset + i] = scale_shape[i];
        }
        for (size_t axis = 0; axis < out_rank; ++axis) {
            if (axis == 1) {
                continue;
            }
            if (aligned_shape[axis] != 1) {
                return false;
            }
        }
        if (aligned_shape[1] != channels || raw_values.size() != channels) {
            return false;
        }
        gamma = std::move(raw_values);
    }

    BatchNormParams params;
    params.gamma = std::move(gamma);
    params.beta.assign(channels, 0.0f);
    params.mean.assign(channels, 0.0f);
    params.var.assign(channels, 1.0f);
    params.epsilon = 0.0f;
    out = std::move(params);
    return true;
}

bool is_attention_group_kind(const std::string& kind) {
    return kind == "Attention" || kind == "AttentionScale" || kind == "AttentionScaleMask";
}

bool is_attention_layout_node(const std::shared_ptr<const ov::Node>& node) {
    return static_cast<bool>(ov::as_type_ptr<const ov::op::v0::Convert>(node)) ||
           static_cast<bool>(ov::as_type_ptr<const ov::op::v1::Broadcast>(node)) ||
           static_cast<bool>(ov::as_type_ptr<const ov::op::v3::Broadcast>(node)) ||
           static_cast<bool>(ov::as_type_ptr<const ov::op::v1::Transpose>(node)) ||
           static_cast<bool>(ov::as_type_ptr<const ov::op::v1::Reshape>(node)) ||
           static_cast<bool>(ov::as_type_ptr<const ov::op::v0::Concat>(node)) ||
           static_cast<bool>(ov::as_type_ptr<const ov::op::v0::Squeeze>(node)) ||
           static_cast<bool>(ov::as_type_ptr<const ov::op::v0::Unsqueeze>(node)) ||
           static_cast<bool>(ov::as_type_ptr<const ov::op::v8::Slice>(node)) ||
           static_cast<bool>(ov::as_type_ptr<const ov::op::v1::StridedSlice>(node)) ||
           static_cast<bool>(ov::as_type_ptr<const ov::op::v1::Split>(node)) ||
           static_cast<bool>(ov::as_type_ptr<const ov::op::v1::VariadicSplit>(node));
}

std::unordered_map<const ov::Node*, std::vector<bool>> collect_model_outputs(const std::shared_ptr<const ov::Model>& model) {
    std::unordered_map<const ov::Node*, std::vector<bool>> model_outputs;
    if (!model) {
        return model_outputs;
    }
    for (const auto& result : model->get_results()) {
        auto src = result->input_value(0).get_node_shared_ptr();
        const size_t port = result->input_value(0).get_index();
        auto& flags = model_outputs[src.get()];
        if (flags.empty()) {
            flags.resize(src->get_output_size(), false);
        }
        if (port < flags.size()) {
            flags[port] = true;
        }
    }
    return model_outputs;
}

bool node_consumed_only_inside_group(
    const std::shared_ptr<const ov::Node>& node,
    const std::unordered_map<const ov::Node*, size_t>& node_index,
    const std::unordered_set<size_t>& group_nodes,
    const std::unordered_map<const ov::Node*, std::vector<bool>>& model_outputs) {
    if (!node || model_outputs.count(node.get()) != 0) {
        return false;
    }
    bool has_internal_consumer = false;
    for (size_t port = 0; port < node->get_output_size(); ++port) {
        const auto& targets = node->output(port).get_target_inputs();
        if (targets.empty()) {
            return false;
        }
        for (const auto& target_input : targets) {
            auto consumer = target_input.get_node()->shared_from_this();
            if (!consumer) {
                return false;
            }
            const auto it = node_index.find(consumer.get());
            if (it == node_index.end() || group_nodes.count(it->second) == 0) {
                return false;
            }
            has_internal_consumer = true;
        }
    }
    return has_internal_consumer;
}

void expand_attention_groups(const std::shared_ptr<const ov::Model>& model, FusionPlan& plan) {
    if (!model || plan.groups.empty()) {
        return;
    }

    const auto ordered_ops = model->get_ordered_ops();
    std::unordered_map<const ov::Node*, size_t> node_index;
    node_index.reserve(ordered_ops.size());
    for (size_t i = 0; i < ordered_ops.size(); ++i) {
        node_index.emplace(ordered_ops[i].get(), i);
    }
    const auto model_outputs = collect_model_outputs(model);

    for (auto& group : plan.groups) {
        if (!is_attention_group_kind(group.kind) || group.node_indices.size() < 3) {
            continue;
        }

        std::unordered_set<size_t> group_nodes(group.node_indices.begin(), group.node_indices.end());

        bool changed = true;
        while (changed) {
            changed = false;
            std::vector<size_t> current(group_nodes.begin(), group_nodes.end());
            std::sort(current.begin(), current.end());
            for (size_t idx : current) {
                if (idx >= ordered_ops.size()) {
                    continue;
                }
                const auto& node = ordered_ops[idx];
                for (const auto& input : node->input_values()) {
                    auto producer = input.get_node_shared_ptr();
                    if (!producer || ov::as_type_ptr<const ov::op::v0::Parameter>(producer) ||
                        ov::as_type_ptr<const ov::op::v0::Constant>(producer) ||
                        !is_attention_layout_node(producer)) {
                        continue;
                    }
                    const auto it = node_index.find(producer.get());
                    if (it == node_index.end() || group_nodes.count(it->second) != 0) {
                        continue;
                    }
                    if (!node_consumed_only_inside_group(producer, node_index, group_nodes, model_outputs)) {
                        continue;
                    }
                    group_nodes.insert(it->second);
                    changed = true;
                }
            }
        }

        size_t tail_idx = *std::max_element(group_nodes.begin(), group_nodes.end());
        while (tail_idx < ordered_ops.size()) {
            const auto& tail = ordered_ops[tail_idx];
            if (tail->get_output_size() != 1) {
                break;
            }
            const auto& targets = tail->output(0).get_target_inputs();
            if (targets.size() != 1) {
                break;
            }
            auto consumer = targets.begin()->get_node()->shared_from_this();
            if (!consumer || ov::as_type_ptr<const ov::op::v0::Result>(consumer) || !is_attention_layout_node(consumer)) {
                break;
            }
            const auto it = node_index.find(consumer.get());
            if (it == node_index.end() || group_nodes.count(it->second) != 0) {
                break;
            }
            group_nodes.insert(it->second);
            tail_idx = it->second;
        }

        group.node_indices.assign(group_nodes.begin(), group_nodes.end());
        std::sort(group.node_indices.begin(), group.node_indices.end());
    }
}

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
    add_conv_batchnorm_swish_fusion_patterns(patterns, config);
    add_conv_batchnorm_act_fusion_patterns(patterns, config);
    add_conv_batchnorm_fusion_patterns(patterns, config);
    add_conv_bias_swish_fusion_patterns(patterns, config);
    add_conv_bias_activation_fusion_patterns(patterns, config);
    add_conv_bias_fusion_patterns(patterns, config);
    add_conv_scale_activation_fusion_patterns(patterns, config);
    add_conv_scale_fusion_patterns(patterns, config);
    add_conv_swish_fusion_patterns(patterns, config);
    add_conv_activation_fusion_patterns(patterns, config);
    add_eltwise_input_activation_fusion_patterns(patterns, config);
    add_eltwise_activation_fusion_patterns(patterns, config);
    add_eltwise_bias_activation_fusion_patterns(patterns, config);
    add_eltwise_bias_fusion_patterns(patterns, config);
    add_matmul_swish_fusion_patterns(patterns, config);
    add_matmul_activation_fusion_patterns(patterns, config);
    add_matmul_bias_swish_fusion_patterns(patterns, config);
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
        if (auto act_attr = op.getAttrOfType<mlir::StringAttr>("gfx.input_activation_kind")) {
            group.input_activation = parse_activation_kind(act_attr.getValue());
        }
        if (auto alpha_attr = op.getAttrOfType<mlir::FloatAttr>("gfx.input_activation_alpha")) {
            group.input_activation_alpha = static_cast<float>(alpha_attr.getValueAsDouble());
        }
        if (auto input_attr = op.getAttrOfType<mlir::IntegerAttr>("gfx.input_activation_input")) {
            group.input_activation_input = static_cast<size_t>(input_attr.getInt());
        }
        if (!group.node_indices.empty()) {
            plan.groups.emplace_back(std::move(group));
        }
    }
    return plan;
}

void materialize_post_op_payloads(const std::shared_ptr<const ov::Model>& model, FusionPlan& plan) {
    if (!model || plan.groups.empty()) {
        return;
    }
    const auto ordered_ops = model->get_ordered_ops();
    for (auto& group : plan.groups) {
        if (group.node_indices.size() < 2) {
            continue;
        }
        const size_t post_op_idx = group.node_indices[1];
        if (post_op_idx >= ordered_ops.size()) {
            continue;
        }
        const size_t primary_idx = group.node_indices.front();
        if (primary_idx >= ordered_ops.size()) {
            continue;
        }
        if (group.kind == "ConvBatchNorm" || group.kind == "ConvBatchNormAct") {
            BatchNormParams params{};
            if (extract_batchnorm_params(ordered_ops[post_op_idx], params)) {
                group.batchnorm = std::move(params);
            }
            continue;
        }
        if (group.kind == "ConvScale" || group.kind == "ConvScaleActivation") {
            BatchNormParams params{};
            if (extract_scale_as_batchnorm_params(ordered_ops[post_op_idx], ordered_ops[primary_idx], params)) {
                group.batchnorm = std::move(params);
            }
            continue;
        }
        if (group.kind == "ConvBias" || group.kind == "ConvBiasActivation" ||
            group.kind == "EltwiseBias" || group.kind == "EltwiseBiasActivation" ||
            group.kind == "MatMulBias" || group.kind == "MatMulBiasActivation") {
            BiasParams params{};
            if (extract_bias_params(ordered_ops[post_op_idx], params)) {
                group.bias = std::move(params);
            }
            continue;
        }
    }
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
    auto& ctx = gfx_mlir_context();
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
    materialize_post_op_payloads(model, plan);
    expand_attention_groups(model, plan);
    return plan;
}

}  // namespace gfx_plugin
}  // namespace ov
