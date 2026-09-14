// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transforms/pipeline.hpp"

#include "runtime/gfx_logger.hpp"
#include "transforms/gfx_layout_cleanup.hpp"
#include "transforms/gfx_llm_ops.hpp"

#include <map>
#include <optional>
#include <functional>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/topk_base.hpp"
#include "openvino/op/variadic_split.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"
#include "transformations/common_optimizations/rms_fusion.hpp"
#include "transformations/control_flow/unroll_if.hpp"
#include "transformations/fp16_compression/convert_compression_only_to_legacy.hpp"
#include "transformations/fp16_compression/mark_subgraphs_to_keep_in_mixed_precision.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/op_conversions/convert_avgpool_downgrade.hpp"
#include "transformations/op_conversions/convert_maxpool_downgrade.hpp"
#include "transformations/op_conversions/convert_mod.hpp"
#include "transformations/op_conversions/convert_reduce_to_pooling.hpp"
#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"

namespace ov {
namespace gfx_plugin {
namespace transforms {
namespace {

struct CompressedMatMulMatch {
    std::shared_ptr<ov::op::v0::MatMul> matmul;
    std::shared_ptr<ov::op::v0::Constant> weights;
    std::shared_ptr<ov::op::v0::Constant> scale;
    ov::Output<ov::Node> data;
    std::vector<std::shared_ptr<ov::Node>> decompression_nodes;
    ov::element::Type weight_input_type = ov::element::dynamic;
    size_t n = 0;
    size_t groups = 0;
    size_t group_size = 0;
    size_t k = 0;
};

struct CompressedMatMulGroupKey {
    const ov::Node* data_node = nullptr;
    size_t data_index = 0;

    bool operator<(const CompressedMatMulGroupKey& other) const {
        return std::tie(data_node, data_index) < std::tie(other.data_node, other.data_index);
    }
};

bool same_value(const ov::Output<ov::Node>& lhs, const ov::Output<ov::Node>& rhs) {
    return lhs.get_node() == rhs.get_node() && lhs.get_index() == rhs.get_index();
}

std::shared_ptr<ov::op::v0::Constant> as_constant(const ov::Output<ov::Node>& value) {
    return ov::as_type_ptr<ov::op::v0::Constant>(value.get_node_shared_ptr());
}

bool has_real_output(const std::shared_ptr<ov::Node>& node) {
    if (!node) {
        return false;
    }
    for (size_t i = 0; i < node->get_output_size(); ++i) {
        if (node->get_output_element_type(i).is_real()) {
            return true;
        }
    }
    return false;
}

bool is_heavy_precision_boundary(const std::shared_ptr<ov::Node>& node) {
    if (!node) {
        return true;
    }
    const std::string type = node->get_type_name();
    return type == "Parameter" ||
           type == "Constant" ||
           type == "Convolution" ||
           type == "GroupConvolution" ||
           type == "MatMul";
}

bool is_conv_precision_boundary(const std::shared_ptr<ov::Node>& node) {
    if (!node) {
        return false;
    }
    const std::string type = node->get_type_name();
    return type == "Convolution" || type == "GroupConvolution";
}

bool is_score_head_data_boundary(const std::shared_ptr<ov::Node>& node) {
    if (!node) {
        return true;
    }
    const std::string type = node->get_type_name();
    return type == "Parameter" ||
           type == "Constant" ||
           type == "Concat" ||
           type == "Split" ||
           type == "VariadicSplit" ||
           type == "MatMul" ||
           type == "ScaledDotProductAttention";
}

size_t real_target_input_count(const ov::Output<ov::Node>& value) {
    size_t count = 0;
    for (const auto& target : value.get_target_inputs()) {
        if (target.get_element_type().is_real()) {
            ++count;
        }
    }
    return count;
}

bool can_extend_fp32_island_through_conv_data_input(
    const ov::Output<ov::Node>& input) {
    if (!input.get_element_type().is_real()) {
        return false;
    }
    auto producer = input.get_node_shared_ptr();
    if (!producer || is_score_head_data_boundary(producer)) {
        return false;
    }
    return real_target_input_count(input) == 1;
}

bool is_topk_precision_sensitive(const std::shared_ptr<ov::Node>& node) {
    auto topk = ov::as_type_ptr<ov::op::util::TopKBase>(node);
    if (!topk || topk->get_input_size() == 0 || !topk->get_input_element_type(0).is_real()) {
        return false;
    }
    if (!topk->get_output_partial_shape(0).is_static() ||
        !topk->get_output_partial_shape(1).is_static()) {
        return true;
    }
    const auto input_shape = topk->get_input_partial_shape(0);
    const int64_t rank = input_shape.rank().is_static() ? input_shape.rank().get_length() : -1;
    const int64_t axis = rank > 0 ? static_cast<int64_t>(topk->get_axis()) : -1;
    const int64_t norm_axis = axis < 0 ? axis + rank : axis;
    return topk->get_k() > 16 ||
           norm_axis < 0 ||
           norm_axis >= rank ||
           topk->get_sort_type() != ov::op::TopKSortType::NONE;
}

bool is_index_amplifying_consumer_input(const std::shared_ptr<ov::Node>& node,
                                        size_t input_index) {
    if (!node) {
        return false;
    }
    const std::string type = node->get_type_name();
    return input_index == 1 &&
           (type == "Gather" ||
            type == "GatherElements" ||
            type == "GatherND" ||
            type == "ScatterElementsUpdate" ||
            type == "ScatterNDUpdate" ||
            type == "ScatterUpdate");
}

bool value_reaches_index_amplifying_consumer(const ov::Output<ov::Node>& value,
                                             std::unordered_set<const ov::Node*>& visited,
                                             size_t depth = 0) {
    if (depth > 64) {
        return false;
    }
    for (const auto& target : value.get_target_inputs()) {
        auto* consumer_ptr = target.get_node();
        if (!consumer_ptr) {
            continue;
        }
        auto consumer = consumer_ptr->shared_from_this();
        if (!consumer || !visited.insert(consumer.get()).second) {
            continue;
        }
        if (is_index_amplifying_consumer_input(consumer, target.get_index())) {
            return true;
        }
        for (size_t output_index = 0; output_index < consumer->get_output_size(); ++output_index) {
            if (value_reaches_index_amplifying_consumer(consumer->output(output_index),
                                                        visited,
                                                        depth + 1)) {
                return true;
            }
        }
    }
    return false;
}

bool topk_indices_are_index_amplified(const std::shared_ptr<ov::Node>& node) {
    auto topk = ov::as_type_ptr<ov::op::util::TopKBase>(node);
    if (!topk || topk->get_output_size() < 2) {
        return false;
    }
    std::unordered_set<const ov::Node*> visited;
    return value_reaches_index_amplifying_consumer(topk->output(1), visited);
}

bool topk_indices_are_observable(const std::shared_ptr<ov::Node>& node) {
    auto topk = ov::as_type_ptr<ov::op::util::TopKBase>(node);
    return topk && topk->get_output_size() >= 2 &&
           !topk->output(1).get_target_inputs().empty();
}

bool topk_values_are_observable(const std::shared_ptr<ov::Node>& node) {
    auto topk = ov::as_type_ptr<ov::op::util::TopKBase>(node);
    return topk && topk->get_output_size() >= 1 &&
           !topk->output(0).get_target_inputs().empty();
}

bool value_reaches_observable_topk_indices(const ov::Output<ov::Node>& value,
                                           std::unordered_set<const ov::Node*>& visited,
                                           size_t depth = 0) {
    if (depth > 64) {
        return false;
    }
    for (const auto& target : value.get_target_inputs()) {
        auto* consumer_ptr = target.get_node();
        if (!consumer_ptr) {
            continue;
        }
        auto consumer = consumer_ptr->shared_from_this();
        if (!consumer || !visited.insert(consumer.get()).second) {
            continue;
        }
        if (target.get_index() == 0 &&
            ov::as_type_ptr<ov::op::util::TopKBase>(consumer) &&
            topk_indices_are_observable(consumer)) {
            return true;
        }
        for (size_t output_index = 0; output_index < consumer->get_output_size(); ++output_index) {
            if (value_reaches_observable_topk_indices(consumer->output(output_index),
                                                      visited,
                                                      depth + 1)) {
                return true;
            }
        }
    }
    return false;
}

bool is_real_sigmoid_output(const ov::Output<ov::Node>& value) {
    if (!value.get_element_type().is_real()) {
        return false;
    }
    return ov::as_type_ptr<ov::op::v0::Sigmoid>(value.get_node_shared_ptr()) != nullptr;
}

std::optional<std::vector<int64_t>> constant_i64_values(const ov::Output<ov::Node>& value) {
    auto constant = as_constant(value);
    if (!constant) {
        return std::nullopt;
    }
    return constant->cast_vector<int64_t>();
}

std::optional<int64_t> constant_i64_scalar(const ov::Output<ov::Node>& value) {
    auto values = constant_i64_values(value);
    if (!values || values->size() != 1) {
        return std::nullopt;
    }
    return values->front();
}

std::optional<int64_t> normalize_axis(int64_t axis, const ov::PartialShape& shape) {
    if (shape.rank().is_dynamic()) {
        return std::nullopt;
    }
    const int64_t rank = shape.rank().get_length();
    if (axis < 0) {
        axis += rank;
    }
    if (axis < 0 || axis >= rank) {
        return std::nullopt;
    }
    return axis;
}

std::optional<ov::Output<ov::Node>> strip_sigmoid_from_split_transpose_concat_path(
    const ov::Output<ov::Node>& value) {
    auto split = ov::as_type_ptr<ov::op::v1::VariadicSplit>(value.get_node_shared_ptr());
    if (!split || split->get_input_size() != 3) {
        return std::nullopt;
    }
    auto split_axis = constant_i64_scalar(split->input_value(1));
    auto split_lengths = constant_i64_values(split->input_value(2));
    if (!split_axis || !split_lengths || value.get_index() >= split_lengths->size()) {
        return std::nullopt;
    }
    auto normalized_split_axis = normalize_axis(*split_axis, split->input_value(0).get_partial_shape());
    if (!normalized_split_axis) {
        return std::nullopt;
    }

    auto concat_value = split->input_value(0);
    std::shared_ptr<ov::op::v1::Transpose> transpose;
    std::optional<std::vector<int64_t>> permutation;
    int64_t concat_axis = *normalized_split_axis;
    if ((transpose = ov::as_type_ptr<ov::op::v1::Transpose>(concat_value.get_node_shared_ptr()))) {
        permutation = constant_i64_values(transpose->input_value(1));
        if (!permutation || *normalized_split_axis >= static_cast<int64_t>(permutation->size())) {
            return std::nullopt;
        }
        concat_axis = (*permutation)[static_cast<size_t>(*normalized_split_axis)];
        concat_value = transpose->input_value(0);
    }

    auto concat = ov::as_type_ptr<ov::op::v0::Concat>(concat_value.get_node_shared_ptr());
    if (!concat) {
        return std::nullopt;
    }
    const auto normalized_concat_axis = normalize_axis(concat->get_axis(), concat_value.get_partial_shape());
    if (!normalized_concat_axis || *normalized_concat_axis != concat_axis ||
        concat->get_input_size() != split_lengths->size()) {
        return std::nullopt;
    }
    for (size_t input_index = 0; input_index < concat->get_input_size(); ++input_index) {
        const auto input_shape = concat->input_value(input_index).get_partial_shape();
        if (input_shape.rank().is_dynamic() ||
            *normalized_concat_axis >= input_shape.rank().get_length() ||
            input_shape[static_cast<size_t>(*normalized_concat_axis)].is_dynamic() ||
            input_shape[static_cast<size_t>(*normalized_concat_axis)].get_length() != (*split_lengths)[input_index]) {
            return std::nullopt;
        }
    }

    auto selected = concat->input_value(value.get_index());
    auto sigmoid = ov::as_type_ptr<ov::op::v0::Sigmoid>(selected.get_node_shared_ptr());
    if (!sigmoid || !selected.get_element_type().is_real()) {
        return std::nullopt;
    }
    auto logits = sigmoid->input_value(0);
    if (!transpose) {
        return logits;
    }

    auto logits_transpose = transpose->clone_with_new_inputs({logits, transpose->input_value(1)});
    logits_transpose->set_friendly_name(transpose->get_friendly_name() + "/gfx_logits");
    ov::copy_runtime_info({sigmoid, transpose, split}, logits_transpose);
    logits_transpose->validate_and_infer_types();
    return logits_transpose->output(0);
}

std::optional<ov::Output<ov::Node>> clone_ranking_path_without_sigmoid(
    const ov::Output<ov::Node>& value,
    size_t depth = 0) {
    if (depth > 32 || !value.get_element_type().is_real()) {
        return std::nullopt;
    }
    if (auto sigmoid = ov::as_type_ptr<ov::op::v0::Sigmoid>(value.get_node_shared_ptr())) {
        return sigmoid->input_value(0);
    }
    if (auto stripped_split = strip_sigmoid_from_split_transpose_concat_path(value)) {
        return stripped_split;
    }

    auto node = value.get_node_shared_ptr();
    if (!node) {
        return std::nullopt;
    }

    auto clone_with_rewritten_data_input =
        [&](size_t data_input_index) -> std::optional<ov::Output<ov::Node>> {
        if (data_input_index >= node->get_input_size()) {
            return std::nullopt;
        }
        auto stripped = clone_ranking_path_without_sigmoid(node->input_value(data_input_index), depth + 1);
        if (!stripped) {
            return std::nullopt;
        }
        ov::OutputVector inputs;
        inputs.reserve(node->get_input_size());
        for (size_t input_index = 0; input_index < node->get_input_size(); ++input_index) {
            inputs.push_back(input_index == data_input_index ? *stripped : node->input_value(input_index));
        }
        auto clone = node->clone_with_new_inputs(inputs);
        clone->set_friendly_name(node->get_friendly_name() + "/gfx_logits");
        ov::copy_runtime_info(node, clone);
        clone->validate_and_infer_types();
        if (value.get_index() >= clone->get_output_size()) {
            return std::nullopt;
        }
        return clone->output(value.get_index());
    };

    const std::string type = node->get_type_name();
    if (type == "ReduceMax" ||
        type == "Reshape" ||
        type == "Transpose" ||
        type == "Split" ||
        type == "VariadicSplit" ||
        type == "Slice" ||
        type == "StridedSlice" ||
        type == "Unsqueeze") {
        return clone_with_rewritten_data_input(0);
    }
    return std::nullopt;
}

size_t canonicalize_sigmoid_before_ranking_ops(const std::shared_ptr<ov::Model>& model) {
    size_t rewrites = 0;
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto& node : model->get_ordered_ops()) {
            if (auto reduce = ov::as_type_ptr<ov::op::v1::ReduceMax>(node)) {
                if (!is_real_sigmoid_output(reduce->input_value(0))) {
                    continue;
                }
                std::unordered_set<const ov::Node*> visited;
                if (value_reaches_observable_topk_indices(reduce->output(0), visited)) {
                    continue;
                }
                auto sigmoid = ov::as_type_ptr<ov::op::v0::Sigmoid>(reduce->input_value(0).get_node_shared_ptr());
                auto logits_reduce =
                    reduce->clone_with_new_inputs({sigmoid->input_value(0), reduce->input_value(1)});
                logits_reduce->set_friendly_name(reduce->get_friendly_name() + "/gfx_logits");
                auto restored_scores = std::make_shared<ov::op::v0::Sigmoid>(logits_reduce);
                restored_scores->set_friendly_name(reduce->get_friendly_name());
                ov::copy_runtime_info({sigmoid, reduce}, logits_reduce);
                ov::copy_runtime_info({sigmoid, reduce}, restored_scores);
                logits_reduce->validate_and_infer_types();
                restored_scores->validate_and_infer_types();
                reduce->output(0).replace(restored_scores->output(0));
                ++rewrites;
                changed = true;
                break;
            }

            auto topk = ov::as_type_ptr<ov::op::util::TopKBase>(node);
            if (!topk || topk->get_output_size() < 2 ||
                !topk->get_output_element_type(0).is_real()) {
                continue;
            }
            const bool values_observable = topk_values_are_observable(topk);
            auto stripped_input = clone_ranking_path_without_sigmoid(topk->input_value(0));
            if (!stripped_input) {
                continue;
            }
            auto logits_topk = topk->clone_with_new_inputs({*stripped_input, topk->input_value(1)});
            logits_topk->set_friendly_name(topk->get_friendly_name() + "/gfx_logits");
            ov::copy_runtime_info(topk, logits_topk);
            logits_topk->validate_and_infer_types();
            if (values_observable) {
                auto restored_values = std::make_shared<ov::op::v0::Sigmoid>(logits_topk->output(0));
                restored_values->set_friendly_name(topk->get_friendly_name());
                ov::copy_runtime_info(topk, restored_values);
                restored_values->validate_and_infer_types();
                topk->output(0).replace(restored_values->output(0));
            } else {
                topk->output(0).replace(logits_topk->output(0));
            }
            topk->output(1).replace(logits_topk->output(1));
            ++rewrites;
            changed = true;
            break;
        }
    }
    return rewrites;
}

bool is_index_selected_data_input(const std::shared_ptr<ov::Node>& node,
                                  size_t input_index) {
    if (!node || input_index != 0) {
        return false;
    }
    const std::string type = node->get_type_name();
    return type == "Gather" ||
           type == "GatherElements" ||
           type == "GatherND" ||
           type == "ScatterElementsUpdate" ||
           type == "ScatterNDUpdate" ||
           type == "ScatterUpdate";
}

void mark_fp32_if_real(const std::shared_ptr<ov::Node>& node, size_t& marked) {
    if (!has_real_output(node) || ov::fp16_compression_is_disabled(node)) {
        return;
    }
    ov::disable_fp16_compression(node);
    ++marked;
}

void mark_terminal_feature_boundary_fp32(
    const ov::Output<ov::Node>& value,
    std::unordered_set<const ov::Node*>& visited,
    std::unordered_set<const ov::Node*>& terminal_boundaries,
    size_t& marked,
    size_t depth = 0) {
    if (depth > 128 || !value.get_element_type().is_real()) {
        return;
    }
    auto node = value.get_node_shared_ptr();
    if (!node || !visited.insert(node.get()).second) {
        return;
    }
    mark_fp32_if_real(node, marked);
    if (is_heavy_precision_boundary(node)) {
        terminal_boundaries.insert(node.get());
        return;
    }
    for (size_t i = 0; i < node->get_input_size(); ++i) {
        const auto input = node->input_value(i);
        if (input.get_element_type().is_real()) {
            mark_terminal_feature_boundary_fp32(input, visited,
                                               terminal_boundaries, marked,
                                               depth + 1);
        }
    }
}

void mark_index_selected_data_inputs_fp32(
    const ov::Output<ov::Node>& value,
    std::unordered_set<const ov::Node*>& visited,
    std::unordered_set<const ov::Node*>& terminal_boundaries,
    size_t& marked,
    size_t depth = 0) {
    if (depth > 64) {
        return;
    }
    for (const auto& target : value.get_target_inputs()) {
        auto* consumer_ptr = target.get_node();
        if (!consumer_ptr) {
            continue;
        }
        auto consumer = consumer_ptr->shared_from_this();
        if (!consumer || !visited.insert(consumer.get()).second) {
            continue;
        }
        if (is_index_amplifying_consumer_input(consumer, target.get_index())) {
            for (size_t input_index = 0; input_index < consumer->get_input_size(); ++input_index) {
                if (!is_index_selected_data_input(consumer, input_index)) {
                    continue;
                }
                const auto data_input = consumer->input_value(input_index);
                if (!data_input.get_element_type().is_real()) {
                    continue;
                }
                std::unordered_set<const ov::Node*> upstream_visited;
                mark_terminal_feature_boundary_fp32(data_input,
                                                   upstream_visited,
                                                   terminal_boundaries,
                                                   marked);
            }
        }
        for (size_t output_index = 0; output_index < consumer->get_output_size(); ++output_index) {
            mark_index_selected_data_inputs_fp32(consumer->output(output_index),
                                                visited, terminal_boundaries,
                                                marked, depth + 1);
        }
    }
}

void mark_topk_upstream_fp32(const ov::Output<ov::Node>& value,
                             std::unordered_set<const ov::Node*>& visited,
                             std::unordered_set<const ov::Node*>& terminal_boundaries,
                             size_t& marked,
                             size_t depth = 0) {
    if (depth > 128) {
        return;
    }
    auto node = value.get_node_shared_ptr();
    if (!node || !visited.insert(node.get()).second) {
        return;
    }
    if (is_conv_precision_boundary(node)) {
        mark_fp32_if_real(node, marked);
        if (node->get_input_size() > 0) {
            const auto data_input = node->input_value(0);
            if (can_extend_fp32_island_through_conv_data_input(data_input)) {
                mark_topk_upstream_fp32(data_input, visited,
                                        terminal_boundaries, marked, depth + 1);
            } else {
                mark_terminal_feature_boundary_fp32(data_input, visited,
                                                   terminal_boundaries, marked,
                                                   depth + 1);
            }
        }
        return;
    }
    mark_fp32_if_real(node, marked);
    if (is_heavy_precision_boundary(node)) {
        return;
    }
    for (size_t i = 0; i < node->get_input_size(); ++i) {
        const auto input = node->input_value(i);
        if (input.get_element_type().is_real()) {
            mark_topk_upstream_fp32(input, visited, terminal_boundaries,
                                    marked, depth + 1);
        }
    }
}

void mark_topk_downstream_fp32(const std::shared_ptr<ov::Node>& node,
                               std::unordered_set<const ov::Node*>& visited,
                               size_t& marked,
                               size_t depth = 0) {
    if (!node || depth > 128) {
        return;
    }
    for (size_t output_index = 0; output_index < node->get_output_size(); ++output_index) {
        for (const auto& target : node->output(output_index).get_target_inputs()) {
            auto consumer = target.get_node()->shared_from_this();
            if (!consumer || !visited.insert(consumer.get()).second) {
                continue;
            }
            mark_fp32_if_real(consumer, marked);
            if (!is_heavy_precision_boundary(consumer)) {
                mark_topk_downstream_fp32(consumer, visited, marked, depth + 1);
            }
        }
    }
}

size_t count_fp32_precision_marks(const std::shared_ptr<ov::Model>& model) {
    size_t count = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::fp16_compression_is_disabled(node)) {
            ++count;
        }
    }
    return count;
}

size_t mark_gfx_order_sensitive_topk_subgraphs(
    const std::shared_ptr<ov::Model>& model,
    std::unordered_set<const ov::Node*>& terminal_boundaries) {
    size_t marked = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (!is_topk_precision_sensitive(node)) {
            continue;
        }
        mark_fp32_if_real(node, marked);
        std::unordered_set<const ov::Node*> upstream_visited;
        const bool index_amplified = topk_indices_are_index_amplified(node);
        mark_topk_upstream_fp32(node->input_value(0), upstream_visited,
                                terminal_boundaries, marked);
        if (index_amplified) {
            std::unordered_set<const ov::Node*> data_visited;
            mark_index_selected_data_inputs_fp32(node->output(1), data_visited,
                                                terminal_boundaries, marked);
        }
        std::unordered_set<const ov::Node*> downstream_visited;
        mark_topk_downstream_fp32(node, downstream_visited, marked);
    }
    return marked;
}

size_t propagate_fp32_marks_upstream_to_boundaries(
    const std::shared_ptr<ov::Model>& model,
    const std::unordered_set<const ov::Node*>& terminal_boundaries) {
    size_t marked = 0;
    constexpr size_t kMaxIterations = 16;
    for (size_t iteration = 0; iteration < kMaxIterations; ++iteration) {
        size_t iteration_marks = 0;
        for (const auto& node : model->get_ordered_ops()) {
            if (!ov::fp16_compression_is_disabled(node)) {
                continue;
            }
            const bool conv_boundary = is_conv_precision_boundary(node);
            if (is_heavy_precision_boundary(node) && !conv_boundary) {
                continue;
            }
            const bool terminal_conv_boundary =
                conv_boundary && terminal_boundaries.count(node.get()) != 0;
            for (size_t i = 0; i < node->get_input_size(); ++i) {
                const auto input = node->input_value(i);
                if (!input.get_element_type().is_real()) {
                    continue;
                }
                if (terminal_conv_boundary && i == 0) {
                    continue;
                }
                auto producer = input.get_node_shared_ptr();
                if (is_conv_precision_boundary(producer)) {
                    if (can_extend_fp32_island_through_conv_data_input(input)) {
                        mark_fp32_if_real(producer, iteration_marks);
                    }
                    if (ov::fp16_compression_is_disabled(producer) &&
                        terminal_boundaries.count(producer.get()) == 0 &&
                        producer->get_input_size() > 0) {
                        const auto data_input = producer->input_value(0);
                        if (can_extend_fp32_island_through_conv_data_input(data_input)) {
                            mark_fp32_if_real(data_input.get_node_shared_ptr(), iteration_marks);
                        }
                    }
                    continue;
                }
                if (conv_boundary && i == 0 &&
                    !can_extend_fp32_island_through_conv_data_input(input)) {
                    continue;
                }
                mark_fp32_if_real(producer, iteration_marks);
            }
        }
        marked += iteration_marks;
        if (iteration_marks == 0) {
            break;
        }
    }
    return marked;
}

size_t apply_gfx_fp32_precision_policy(const std::shared_ptr<ov::Model>& model) {
    const auto before = count_fp32_precision_marks(model);

    ov::pass::Manager precision_manager("Plugin:GFX:PrecisionPolicy");
    precision_manager.register_pass<ov::pass::MarkSugraphsToKeepInMixedPrecision>();
    precision_manager.run_passes(model);

    const auto after_common = count_fp32_precision_marks(model);
    std::unordered_set<const ov::Node*> terminal_boundaries;
    const auto gfx_topk_marks =
        mark_gfx_order_sensitive_topk_subgraphs(model, terminal_boundaries);
    const auto propagated_marks =
        propagate_fp32_marks_upstream_to_boundaries(model, terminal_boundaries);
    return (after_common - before) + gfx_topk_marks + propagated_marks;
}

bool is_scalar_like(const ov::Output<ov::Node>& value) {
    const auto pshape = value.get_partial_shape();
    if (pshape.rank().is_dynamic()) {
        return false;
    }
    if (pshape.rank().get_length() == 0) {
        return true;
    }
    if (pshape.is_static()) {
        return ov::shape_size(pshape.to_shape()) == 1;
    }
    return false;
}

std::optional<ov::Output<ov::Node>> slice_data_input(const ov::Output<ov::Node>& value) {
    const auto node = value.get_node_shared_ptr();
    if (auto slice = ov::as_type_ptr<ov::op::v8::Slice>(node)) {
        return slice->input_value(0);
    }
    if (auto strided = ov::as_type_ptr<ov::op::v1::StridedSlice>(node)) {
        return strided->input_value(0);
    }
    return std::nullopt;
}

std::optional<ov::Output<ov::Node>> negated_slice_data_input(const ov::Output<ov::Node>& value) {
    auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(value.get_node_shared_ptr());
    if (!mul || mul->get_input_size() != 2) {
        return std::nullopt;
    }
    for (size_t idx = 0; idx < 2; ++idx) {
        auto maybe_slice = slice_data_input(mul->input_value(idx));
        if (maybe_slice && is_scalar_like(mul->input_value(1 - idx))) {
            return maybe_slice;
        }
    }
    return std::nullopt;
}

struct RotateHalfMatch {
    ov::Output<ov::Node> data;
};

std::optional<RotateHalfMatch> match_rotate_half_concat(const ov::Output<ov::Node>& value) {
    auto concat = ov::as_type_ptr<ov::op::v0::Concat>(value.get_node_shared_ptr());
    if (!concat || concat->get_input_size() != 2) {
        return std::nullopt;
    }
    const auto out_pshape = concat->get_output_partial_shape(0);
    if (out_pshape.rank().is_dynamic() || out_pshape.rank().get_length() == 0 ||
        out_pshape[out_pshape.rank().get_length() - 1].is_dynamic()) {
        return std::nullopt;
    }
    const auto rank = static_cast<size_t>(out_pshape.rank().get_length());
    const auto axis = concat->get_axis() < 0
                          ? static_cast<int64_t>(rank) + concat->get_axis()
                          : concat->get_axis();
    if (axis != static_cast<int64_t>(rank - 1)) {
        return std::nullopt;
    }
    const auto head_size = static_cast<size_t>(out_pshape[rank - 1].get_length());
    if (head_size == 0 || (head_size % 2) != 0) {
        return std::nullopt;
    }
    const auto first_pshape = concat->get_input_partial_shape(0);
    const auto second_pshape = concat->get_input_partial_shape(1);
    if (first_pshape.rank().is_dynamic() || second_pshape.rank().is_dynamic() ||
        first_pshape.rank().get_length() != static_cast<int64_t>(rank) ||
        second_pshape.rank().get_length() != static_cast<int64_t>(rank) ||
        first_pshape[rank - 1].is_dynamic() || second_pshape[rank - 1].is_dynamic() ||
        static_cast<size_t>(first_pshape[rank - 1].get_length()) != head_size / 2 ||
        static_cast<size_t>(second_pshape[rank - 1].get_length()) != head_size / 2) {
        return std::nullopt;
    }

    auto neg_src = negated_slice_data_input(concat->input_value(0));
    auto pos_src = slice_data_input(concat->input_value(1));
    if (!neg_src || !pos_src || !same_value(*neg_src, *pos_src)) {
        return std::nullopt;
    }
    return RotateHalfMatch{*neg_src};
}

std::optional<ov::Output<ov::Node>> non_matching_multiply_input(const std::shared_ptr<ov::op::v1::Multiply>& mul,
                                                                const ov::Output<ov::Node>& match) {
    if (!mul || mul->get_input_size() != 2) {
        return std::nullopt;
    }
    if (same_value(mul->input_value(0), match)) {
        return mul->input_value(1);
    }
    if (same_value(mul->input_value(1), match)) {
        return mul->input_value(0);
    }
    return std::nullopt;
}

bool is_integer_element_type(const ov::element::Type& type) {
    return type == ov::element::i8 || type == ov::element::u8 ||
           type == ov::element::i16 || type == ov::element::u16 ||
           type == ov::element::i32 || type == ov::element::u32 ||
           type == ov::element::i64 || type == ov::element::u64;
}

bool is_sdpa_causal_mask_abi_integer_type(const ov::element::Type& type) {
    return type == ov::element::i64;
}

bool dimension_compatible(const ov::Dimension& lhs, const ov::Dimension& rhs) {
    return lhs.is_dynamic() || rhs.is_dynamic() || lhs.get_length() == rhs.get_length();
}

bool is_shape_metadata_value(const ov::Output<ov::Node>& value) {
    const auto node = value.get_node_shared_ptr();
    return ov::as_type_ptr<ov::op::v0::Constant>(node) ||
           ov::as_type_ptr<ov::op::v0::ShapeOf>(node) ||
           ov::as_type_ptr<ov::op::v3::ShapeOf>(node);
}

struct UpstreamValueCandidate {
    ov::Output<ov::Node> value;
    size_t depth = 0;
    int score = 0;
};

std::optional<ov::Output<ov::Node>> find_best_upstream_value(
    const ov::Output<ov::Node>& root,
    const std::function<std::optional<int>(const ov::Output<ov::Node>&)>& scorer,
    size_t max_depth = 64) {
    struct Item {
        ov::Output<ov::Node> value;
        size_t depth = 0;
    };
    struct Key {
        const ov::Node* node = nullptr;
        size_t port = 0;
        bool operator==(const Key& other) const {
            return node == other.node && port == other.port;
        }
    };
    struct KeyHash {
        size_t operator()(const Key& key) const {
            const size_t h1 = std::hash<const ov::Node*>()(key.node);
            const size_t h2 = std::hash<size_t>()(key.port);
            return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
        }
    };

    std::vector<Item> stack{{root, 0}};
    std::unordered_set<Key, KeyHash> visited;
    std::optional<UpstreamValueCandidate> best;
    while (!stack.empty()) {
        auto item = stack.back();
        stack.pop_back();
        const Key key{item.value.get_node(), item.value.get_index()};
        if (!visited.insert(key).second) {
            continue;
        }
        if (auto score = scorer(item.value)) {
            if (!best ||
                *score > best->score ||
                (*score == best->score && item.depth < best->depth)) {
                best = UpstreamValueCandidate{item.value, item.depth, *score};
            }
        }
        if (item.depth >= max_depth) {
            continue;
        }
        auto node = item.value.get_node_shared_ptr();
        if (!node) {
            continue;
        }
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            stack.push_back({node->input_value(i), item.depth + 1});
        }
    }
    if (best) {
        return best->value;
    }
    return std::nullopt;
}

std::optional<ov::Output<ov::Node>> find_llm_attention_mask_input(const ov::Output<ov::Node>& mask_value,
                                                                  const ov::PartialShape& q_shape,
                                                                  const ov::PartialShape& k_shape) {
    return find_best_upstream_value(mask_value, [&](const ov::Output<ov::Node>& value) -> std::optional<int> {
        if (is_shape_metadata_value(value)) {
            return std::nullopt;
        }
        const auto type = value.get_element_type();
        const auto pshape = value.get_partial_shape();
        const auto rank = pshape.rank();
        if (!is_sdpa_causal_mask_abi_integer_type(type) || rank.is_dynamic() || rank.get_length() != 2) {
            return std::nullopt;
        }
        if (q_shape.rank().is_static() && q_shape.rank().get_length() == 4 &&
            !dimension_compatible(pshape[0], q_shape[0])) {
            return std::nullopt;
        }
        if (k_shape.rank().is_static() && k_shape.rank().get_length() == 4 &&
            !dimension_compatible(pshape[1], k_shape[2])) {
            return std::nullopt;
        }
        int score = 10;
        if (ov::as_type_ptr<ov::op::v0::Parameter>(value.get_node_shared_ptr())) {
            score += 100;
        }
        if (pshape.is_static()) {
            score += 5;
        }
        return score;
    });
}

std::optional<ov::Output<ov::Node>> find_llm_cache_positions(const ov::Output<ov::Node>& mask_value,
                                                            const ov::PartialShape& q_shape) {
    if (q_shape.rank().is_dynamic() || q_shape.rank().get_length() != 4 || q_shape[2].is_dynamic()) {
        return std::nullopt;
    }
    return find_best_upstream_value(mask_value, [&](const ov::Output<ov::Node>& value) -> std::optional<int> {
        if (is_shape_metadata_value(value)) {
            return std::nullopt;
        }
        const auto type = value.get_element_type();
        const auto pshape = value.get_partial_shape();
        const auto rank = pshape.rank();
        if (!is_sdpa_causal_mask_abi_integer_type(type) || rank.is_dynamic() || rank.get_length() != 1) {
            return std::nullopt;
        }
        if (!dimension_compatible(pshape[0], q_shape[2])) {
            return std::nullopt;
        }
        int score = 10;
        if (ov::as_type_ptr<ov::op::v0::Parameter>(value.get_node_shared_ptr())) {
            score += 100;
        }
        if (pshape.is_static()) {
            score += 5;
        }
        return score;
    });
}

std::optional<ov::Output<ov::Node>> peel_llm_gqa_broadcast_view(const ov::Output<ov::Node>& value) {
    auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(value.get_node_shared_ptr());
    if (!reshape || reshape->get_input_size() != 2) {
        return std::nullopt;
    }
    auto broadcast_node = reshape->input_value(0).get_node_shared_ptr();
    if (!ov::as_type_ptr<ov::op::v1::Broadcast>(broadcast_node) &&
        !ov::as_type_ptr<ov::op::v3::Broadcast>(broadcast_node)) {
        return std::nullopt;
    }
    auto unsqueeze = ov::as_type_ptr<ov::op::v0::Unsqueeze>(broadcast_node->input_value(0).get_node_shared_ptr());
    if (!unsqueeze || unsqueeze->get_input_size() != 2) {
        return std::nullopt;
    }

    auto axes = as_constant(unsqueeze->input_value(1));
    if (!axes) {
        return std::nullopt;
    }
    const auto axes_values = axes->cast_vector<int64_t>();
    if (axes_values.size() != 1 || axes_values.front() != 2) {
        return std::nullopt;
    }

    const auto compact = unsqueeze->input_value(0);
    const auto compact_pshape = compact.get_partial_shape();
    const auto unsqueeze_pshape = unsqueeze->get_output_partial_shape(0);
    const auto broadcast_pshape = broadcast_node->get_output_partial_shape(0);
    const auto reshape_pshape = reshape->get_output_partial_shape(0);
    if (compact_pshape.rank().is_dynamic() || unsqueeze_pshape.rank().is_dynamic() ||
        broadcast_pshape.rank().is_dynamic() || reshape_pshape.rank().is_dynamic() ||
        compact_pshape.rank().get_length() != 4 ||
        unsqueeze_pshape.rank().get_length() != 5 ||
        broadcast_pshape.rank().get_length() != 5 ||
        reshape_pshape.rank().get_length() != 4) {
        return std::nullopt;
    }
    if (compact_pshape[1].is_static() && reshape_pshape[1].is_static() &&
        compact_pshape[1].get_length() >= reshape_pshape[1].get_length()) {
        return std::nullopt;
    }
    if (compact_pshape[3].is_static() && reshape_pshape[3].is_static() &&
        compact_pshape[3].get_length() != reshape_pshape[3].get_length()) {
        return std::nullopt;
    }
    return compact;
}

size_t fuse_llm_sdpa_causal_mask(const std::shared_ptr<ov::Model>& model) {
    size_t fused = 0;
    size_t peeled_gqa = 0;
    for (const auto& node : model->get_ordered_ops()) {
        auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(node);
        if (!sdpa || sdpa->get_input_size() != 5 || sdpa->get_output_size() != 1) {
            continue;
        }
        auto attention_mask = find_llm_attention_mask_input(sdpa->input_value(3),
                                                            sdpa->input_value(0).get_partial_shape(),
                                                            sdpa->input_value(1).get_partial_shape());
        auto cache_positions = find_llm_cache_positions(sdpa->input_value(3),
                                                        sdpa->input_value(0).get_partial_shape());
        if (!attention_mask || !cache_positions) {
            continue;
        }

        auto k_input = sdpa->input_value(1);
        auto v_input = sdpa->input_value(2);
        if (auto compact_k = peel_llm_gqa_broadcast_view(k_input)) {
            k_input = *compact_k;
            ++peeled_gqa;
        }
        if (auto compact_v = peel_llm_gqa_broadcast_view(v_input)) {
            v_input = *compact_v;
            ++peeled_gqa;
        }

        ov::OutputVector inputs{
            sdpa->input_value(0),
            k_input,
            v_input,
            *attention_mask,
            *cache_positions,
            sdpa->input_value(4),
        };
        auto fused_sdpa = std::make_shared<ov::gfx_plugin::op::GfxSDPAWithCausalMask>(inputs);
        fused_sdpa->set_friendly_name(sdpa->get_friendly_name() + "/gfx_causal_mask");
        ov::copy_runtime_info({sdpa, sdpa->input_value(3).get_node_shared_ptr()}, fused_sdpa);
        fused_sdpa->validate_and_infer_types();
        sdpa->output(0).replace(fused_sdpa->output(0));
        ++fused;
    }

    if (gfx_log_debug_enabled()) {
        gfx_log_debug("GfxTransforms") << "LLaMA SDPA causal-mask fusion: fused=" << fused
                                       << " peeled_gqa=" << peeled_gqa;
    }
    return fused;
}

std::optional<CompressedMatMulMatch> match_compressed_matmul(const std::shared_ptr<ov::Node>& node) {
    auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(node);
    if (!matmul || !matmul->get_transpose_b() || matmul->get_input_size() != 2 ||
        matmul->get_output_size() != 1) {
        return std::nullopt;
    }
    if (!matmul->get_input_partial_shape(1).is_static()) {
        return std::nullopt;
    }
    const auto b_shape = matmul->get_input_shape(1);
    if (b_shape.size() != 2) {
        return std::nullopt;
    }

    CompressedMatMulMatch match;
    auto source = matmul->input_value(1).get_node_shared_ptr();
    const auto weight_input_type = matmul->get_input_element_type(1);
    if (auto convert = ov::as_type_ptr<ov::op::v0::Convert>(source)) {
        match.decompression_nodes.push_back(convert);
        source = convert->input_value(0).get_node_shared_ptr();
    }
    auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(source);
    if (!reshape) {
        return std::nullopt;
    }
    match.decompression_nodes.push_back(reshape);
    auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(reshape->input_value(0).get_node_shared_ptr());
    if (!mul) {
        return std::nullopt;
    }
    match.decompression_nodes.push_back(mul);

    std::shared_ptr<ov::op::v0::Constant> weights;
    std::shared_ptr<ov::op::v0::Constant> scale;
    for (size_t i = 0; i < mul->get_input_size(); ++i) {
        auto input = mul->input_value(i);
        if (auto convert = ov::as_type_ptr<ov::op::v0::Convert>(input.get_node_shared_ptr())) {
            match.decompression_nodes.push_back(convert);
            if (auto constant = as_constant(convert->input_value(0))) {
                const auto et = constant->get_element_type();
                if (et == ov::element::i4 || et == ov::element::u4 ||
                    et == ov::element::i8 || et == ov::element::u8) {
                    weights = constant;
                    continue;
                }
            }
        }
        if (auto constant = as_constant(input)) {
            if (constant->get_element_type() == ov::element::f16 ||
                constant->get_element_type() == ov::element::f32) {
                scale = constant;
            }
        }
    }
    if (!weights || !scale) {
        return std::nullopt;
    }

    const auto raw_shape = weights->get_shape();
    const auto scale_shape = scale->get_shape();
    if (raw_shape.size() != 3 || scale_shape.size() != 3 ||
        raw_shape[0] != b_shape[0] ||
        scale_shape[0] != raw_shape[0] ||
        scale_shape[1] != raw_shape[1] ||
        scale_shape[2] != 1) {
        return std::nullopt;
    }
    const size_t n = raw_shape[0];
    const size_t groups = raw_shape[1];
    const size_t group_size = raw_shape[2];
    const size_t k = groups * group_size;
    if (n == 0 || k == 0 || b_shape[1] != k) {
        return std::nullopt;
    }

    match.matmul = matmul;
    match.weights = weights;
    match.scale = scale;
    match.data = matmul->input_value(0);
    match.weight_input_type = weight_input_type;
    match.n = n;
    match.groups = groups;
    match.group_size = group_size;
    match.k = k;
    return match;
}

void protect_compressed_decompression(const CompressedMatMulMatch& match) {
    for (const auto& node : match.decompression_nodes) {
        ov::mark_as_decompression(node);
        ov::disable_constant_folding(node);
    }
}

void protect_compressed_matmul_decompressions(const std::shared_ptr<ov::Model>& model) {
    size_t match_count = 0;
    size_t matmul_count = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (node->get_type_name() == std::string("MatMul")) {
            ++matmul_count;
        }
        auto match = match_compressed_matmul(node);
        if (!match) {
            continue;
        }
        protect_compressed_decompression(*match);
        ++match_count;
    }

    if (gfx_log_debug_enabled()) {
        gfx_log_debug("GfxTransforms") << "compressed MatMul decompression protection: matmuls=" << matmul_count
                                       << " matches=" << match_count;
    }
}

bool compatible_for_horizontal_fusion(const CompressedMatMulMatch& base,
                                      const CompressedMatMulMatch& candidate,
                                      size_t output_rank) {
    if (candidate.data.get_node() != base.data.get_node() ||
        candidate.data.get_index() != base.data.get_index() ||
        candidate.matmul->get_transpose_a() != base.matmul->get_transpose_a() ||
        candidate.weight_input_type != base.weight_input_type ||
        candidate.weights->get_element_type() != base.weights->get_element_type() ||
        candidate.scale->get_element_type() != base.scale->get_element_type() ||
        candidate.groups != base.groups ||
        candidate.group_size != base.group_size ||
        candidate.k != base.k) {
        return false;
    }
    const auto output_pshape = candidate.matmul->get_output_partial_shape(0);
    if (output_pshape.rank().is_dynamic() ||
        static_cast<size_t>(output_pshape.rank().get_length()) != output_rank ||
        output_pshape[output_rank - 1].is_dynamic() ||
        static_cast<size_t>(output_pshape[output_rank - 1].get_length()) != candidate.n) {
        return false;
    }
    return true;
}

size_t fuse_compressed_matmul_horizontal(const std::shared_ptr<ov::Model>& model) {
    std::map<CompressedMatMulGroupKey, std::vector<CompressedMatMulMatch>> groups;
    for (const auto& node : model->get_ordered_ops()) {
        auto match = match_compressed_matmul(node);
        if (!match) {
            continue;
        }
        const CompressedMatMulGroupKey key{match->data.get_node(), match->data.get_index()};
        groups[key].push_back(std::move(*match));
    }

    size_t fused_groups = 0;
    size_t fused_matmuls = 0;
    for (auto& entry : groups) {
        auto& matches = entry.second;
        if (matches.size() < 2) {
            continue;
        }

        const auto output_pshape = matches.front().matmul->get_output_partial_shape(0);
        if (output_pshape.rank().is_dynamic()) {
            continue;
        }
        const size_t output_rank = static_cast<size_t>(output_pshape.rank().get_length());
        if (output_rank == 0 ||
            output_pshape[output_rank - 1].is_dynamic() ||
            static_cast<size_t>(output_pshape[output_rank - 1].get_length()) != matches.front().n) {
            continue;
        }

        bool compatible = true;
        size_t total_n = 0;
        std::vector<ov::Output<ov::Node>> weight_inputs;
        std::vector<int64_t> split_lengths;
        ov::NodeVector rt_sources;
        weight_inputs.reserve(matches.size());
        split_lengths.reserve(matches.size());
        for (const auto& match : matches) {
            if (!compatible_for_horizontal_fusion(matches.front(), match, output_rank)) {
                compatible = false;
                break;
            }
            total_n += match.n;
            weight_inputs.push_back(match.matmul->input_value(1));
            split_lengths.push_back(static_cast<int64_t>(match.n));
            rt_sources.push_back(match.matmul);
            for (const auto& node : match.decompression_nodes) {
                rt_sources.push_back(node);
            }
        }
        if (!compatible || total_n == 0) {
            continue;
        }

        auto concat = std::make_shared<ov::op::v0::Concat>(weight_inputs, 0);
        concat->set_friendly_name(matches.front().matmul->get_friendly_name() + "/gfx_fused_weight_concat");
        ov::mark_as_decompression(concat);
        ov::disable_constant_folding(concat);

        auto fused_matmul = std::make_shared<ov::op::v0::MatMul>(matches.front().data,
                                                                 concat,
                                                                 matches.front().matmul->get_transpose_a(),
                                                                 true);
        fused_matmul->set_friendly_name(matches.front().matmul->get_friendly_name() + "/gfx_fused_horizontal");

        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {static_cast<int64_t>(output_rank - 1)});
        auto lengths = ov::op::v0::Constant::create(ov::element::i64,
                                                    ov::Shape{split_lengths.size()},
                                                    split_lengths);
        auto split = std::make_shared<ov::op::v1::VariadicSplit>(fused_matmul, axis, lengths);
        split->set_friendly_name(matches.front().matmul->get_friendly_name() + "/gfx_fused_horizontal_split");
        ov::copy_runtime_info(rt_sources, ov::NodeVector{concat, fused_matmul, split});

        split->validate_and_infer_types();
        for (size_t i = 0; i < matches.size(); ++i) {
            matches[i].matmul->output(0).replace(split->output(i));
        }
        ++fused_groups;
        fused_matmuls += matches.size();
    }

    if (gfx_log_debug_enabled()) {
        gfx_log_debug("GfxTransforms") << "compressed MatMul horizontal fusion: groups=" << fused_groups
                                       << " matmuls=" << fused_matmuls;
    }
    return fused_groups;
}

size_t fuse_llama_rotate_half_rope(const std::shared_ptr<ov::Model>& model) {
    size_t fused = 0;
    for (const auto& node : model->get_ordered_ops()) {
        auto add = ov::as_type_ptr<ov::op::v1::Add>(node);
        if (!add || add->get_input_size() != 2 || add->get_output_size() != 1) {
            continue;
        }
        auto mul0 = ov::as_type_ptr<ov::op::v1::Multiply>(add->input_value(0).get_node_shared_ptr());
        auto mul1 = ov::as_type_ptr<ov::op::v1::Multiply>(add->input_value(1).get_node_shared_ptr());
        if (!mul0 || !mul1) {
            continue;
        }

        std::shared_ptr<ov::op::v1::Multiply> direct_mul;
        std::shared_ptr<ov::op::v1::Multiply> rotated_mul;
        std::optional<RotateHalfMatch> rotate;
        std::optional<ov::Output<ov::Node>> rotate_input;
        for (size_t idx = 0; idx < 2; ++idx) {
            auto candidate_rotated = idx == 0 ? mul0 : mul1;
            for (size_t input_idx = 0; input_idx < 2; ++input_idx) {
                auto candidate = match_rotate_half_concat(candidate_rotated->input_value(input_idx));
                if (!candidate) {
                    continue;
                }
                direct_mul = idx == 0 ? mul1 : mul0;
                rotated_mul = candidate_rotated;
                rotate = candidate;
                rotate_input = candidate_rotated->input_value(input_idx);
                break;
            }
            if (rotate) {
                break;
            }
        }
        if (!rotate || !rotate_input) {
            continue;
        }

        auto cos = non_matching_multiply_input(direct_mul, rotate->data);
        auto sin = non_matching_multiply_input(rotated_mul, *rotate_input);
        if (!cos || !sin) {
            continue;
        }
        const auto data_pshape = rotate->data.get_partial_shape();
        if (data_pshape.rank().is_dynamic() || data_pshape.rank().get_length() < 2 ||
            data_pshape[data_pshape.rank().get_length() - 1].is_dynamic()) {
            continue;
        }
        const auto head_size = static_cast<size_t>(data_pshape[data_pshape.rank().get_length() - 1].get_length());
        if (head_size == 0 || (head_size % 2) != 0) {
            continue;
        }

        ov::op::internal::RoPE::Config cfg;
        cfg.rotary_ndims = head_size;
        cfg.cos_sin_ndims = head_size;
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{rotate->data, *cos, *sin}, cfg);
        rope->set_friendly_name(add->get_friendly_name() + "/gfx_rope");
        ov::copy_runtime_info({add, direct_mul, rotated_mul, rotate_input->get_node_shared_ptr()}, rope);
        rope->validate_and_infer_types();
        add->output(0).replace(rope->output(0));
        ++fused;
    }

    if (gfx_log_debug_enabled()) {
        gfx_log_debug("GfxTransforms") << "LLaMA rotate-half RoPE fusion: fused=" << fused;
    }
    return fused;
}

}  // namespace

std::shared_ptr<const ov::Model> run_pipeline(const std::shared_ptr<const ov::Model>& model,
                                              const PipelineOptions& options) {
    OPENVINO_ASSERT(model, "run_pipeline: model is null");
    // Work on a clone to preserve the original model passed by the caller.
    auto cloned = model->clone();

    ov::pass::Manager manager("Plugin:GFX");
    // Common optimizations from OpenVINO transformations library.
    manager.register_pass<ov::pass::CommonOptimizations>();
    // LLM RMSNorm frequently arrives as Power->ReduceMean->Add->Sqrt/Divide->Multiply->Multiply.
    // Fuse both no-tail-convert and Divide variants into one backend-lowerable op.
    manager.register_pass<ov::pass::RMSFusion>(false, true);
    // Plugin-local structural cleanup before stage selection / MLIR lowering.
    manager.register_pass<ov::gfx_plugin::transforms::GfxLayoutCleanup>();

    // Align behaviour with template plugin: disable a few transformations that can be harmful for backend mapping.
    auto pass_config = manager.get_pass_config();
    pass_config->disable<ov::pass::UnrollIf>();
    pass_config->disable<ov::pass::ConvertMaxPool14ToMaxPool8>();
    pass_config->disable<ov::pass::ConvertAvgPool14ToAvgPool1>();
    pass_config->disable<ov::pass::ConvertReduceSumToPooling>();
    pass_config->disable<ov::pass::ConvertMod>();
    if (options.preserve_scaled_dot_product_attention) {
        pass_config->disable<ov::pass::ScaledDotProductAttentionDecomposition>();
    }

    // Keep compressed weight decompression subgraphs intact for backend-side weight-only kernels.
    pass_config->disable<ov::pass::ConvertCompressedOnlyToLegacy>();

    const bool canonicalize_sigmoid_ranking =
        options.canonicalize_sigmoid_before_ranking;
    size_t sigmoid_ranking_rewrites = 0;
    if (canonicalize_sigmoid_ranking) {
        sigmoid_ranking_rewrites = canonicalize_sigmoid_before_ranking_ops(cloned);
    }
    protect_compressed_matmul_decompressions(cloned);
    cloned->validate_nodes_and_infer_types();
    manager.run_passes(cloned);
    if (canonicalize_sigmoid_ranking) {
        sigmoid_ranking_rewrites += canonicalize_sigmoid_before_ranking_ops(cloned);
    }
    if (gfx_log_debug_enabled() && sigmoid_ranking_rewrites != 0) {
        gfx_log_debug("GfxTransforms") << "Canonicalized sigmoid ranking ops="
                                       << sigmoid_ranking_rewrites;
    }
    const auto precision_sensitive_count = apply_gfx_fp32_precision_policy(cloned);
    if (gfx_log_debug_enabled() && precision_sensitive_count != 0) {
        gfx_log_debug("GfxTransforms") << "Marked fp32 precision-sensitive nodes="
                                       << precision_sensitive_count;
    }
    if (options.enable_llm_attention_fusions) {
        fuse_llama_rotate_half_rope(cloned);
        fuse_llm_sdpa_causal_mask(cloned);
    }
    fuse_compressed_matmul_horizontal(cloned);
    cloned->validate_nodes_and_infer_types();
    return cloned;
}

}  // namespace transforms
}  // namespace gfx_plugin
}  // namespace ov
