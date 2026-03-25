// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transforms/gfx_layout_cleanup.hpp"

#include <numeric>
#include <optional>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace gfx_plugin {
namespace transforms {
namespace {

bool is_identity_permutation(const std::shared_ptr<const ov::Node>& perm_source,
                             size_t expected_rank) {
    auto perm_const = ov::as_type_ptr<const ov::op::v0::Constant>(perm_source);
    if (!perm_const) {
        return false;
    }
    const auto perm = perm_const->cast_vector<int64_t>();
    if (perm.size() != expected_rank) {
        return false;
    }
    for (size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] != static_cast<int64_t>(i)) {
            return false;
        }
    }
    return true;
}

std::optional<std::vector<int64_t>> get_constant_permutation(const ov::Output<ov::Node>& perm_source) {
    auto perm_const = ov::as_type_ptr<const ov::op::v0::Constant>(perm_source.get_node_shared_ptr());
    if (!perm_const) {
        return std::nullopt;
    }
    return perm_const->cast_vector<int64_t>();
}

bool is_valid_permutation(const std::vector<int64_t>& perm) {
    std::vector<bool> seen(perm.size(), false);
    for (int64_t axis : perm) {
        if (axis < 0 || axis >= static_cast<int64_t>(perm.size())) {
            return false;
        }
        if (seen[static_cast<size_t>(axis)]) {
            return false;
        }
        seen[static_cast<size_t>(axis)] = true;
    }
    return true;
}

bool is_identity_permutation_vector(const std::vector<int64_t>& perm) {
    for (size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] != static_cast<int64_t>(i)) {
            return false;
        }
    }
    return true;
}

std::vector<int64_t> compose_permutations(const std::vector<int64_t>& first,
                                          const std::vector<int64_t>& second) {
    std::vector<int64_t> composed(second.size(), 0);
    for (size_t i = 0; i < second.size(); ++i) {
        composed[i] = first[static_cast<size_t>(second[i])];
    }
    return composed;
}

std::shared_ptr<ov::op::v0::Constant> make_i64_constant(const std::vector<int64_t>& values) {
    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
}

std::shared_ptr<ov::op::v0::Constant> get_constant_from_source_local(const ov::Output<ov::Node>& source) {
    auto node = source.get_node_shared_ptr();
    if (!node) {
        return nullptr;
    }
    if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node)) {
        return constant;
    }
    if (!node->has_evaluate()) {
        return nullptr;
    }
    ov::TensorVector inputs;
    inputs.reserve(node->get_input_size());
    for (const auto& input_value : node->input_values()) {
        auto input_constant = get_constant_from_source_local(input_value);
        if (!input_constant) {
            return nullptr;
        }
        inputs.push_back(input_constant->get_tensor_view());
    }
    ov::TensorVector outputs;
    outputs.reserve(node->get_output_size());
    for (size_t i = 0; i < node->get_output_size(); ++i) {
        if (node->get_output_partial_shape(i).is_dynamic()) {
            return nullptr;
        }
        outputs.emplace_back(node->get_output_element_type(i), node->get_output_shape(i));
    }
    if (!node->evaluate(outputs, inputs) || outputs.empty()) {
        return nullptr;
    }
    return std::make_shared<ov::op::v0::Constant>(outputs.front());
}

std::optional<std::vector<int64_t>> get_constant_i64_values(const ov::Output<ov::Node>& source) {
    const auto constant = get_constant_from_source_local(source);
    if (!constant || constant->get_element_type() != ov::element::i64) {
        return std::nullopt;
    }
    return constant->cast_vector<int64_t>();
}

std::optional<int64_t> perfect_square_root(int64_t value) {
    if (value <= 0) {
        return std::nullopt;
    }
    int64_t root = 1;
    while (root * root < value) {
        ++root;
    }
    if (root * root != value) {
        return std::nullopt;
    }
    return root;
}

bool eliminate_identity_transpose(const std::shared_ptr<ov::Node>& node) {
    auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(node);
    if (!transpose || transpose->get_input_size() != 2 || transpose->get_output_size() != 1) {
        return false;
    }
    const auto& in_pshape = transpose->get_input_partial_shape(0);
    const auto& out_pshape = transpose->get_output_partial_shape(0);
    if (in_pshape.is_dynamic() || out_pshape.is_dynamic() || in_pshape != out_pshape) {
        return false;
    }
    if (!is_identity_permutation(transpose->input_value(1).get_node_shared_ptr(), in_pshape.rank().get_length())) {
        return false;
    }
    return ov::replace_output_update_name(transpose->output(0), transpose->input_value(0));
}

bool fold_transpose_softmax_transpose(const std::shared_ptr<ov::Node>& node) {
    auto transpose_after = ov::as_type_ptr<ov::op::v1::Transpose>(node);
    if (!transpose_after || transpose_after->get_input_size() != 2 || transpose_after->get_output_size() != 1) {
        return false;
    }

    auto softmax = transpose_after->input_value(0).get_node_shared_ptr();
    if (!softmax || softmax->get_output_size() != 1 || softmax->output(0).get_target_inputs().size() != 1) {
        return false;
    }
    const bool is_softmax_v8 = static_cast<bool>(ov::as_type_ptr<ov::op::v8::Softmax>(softmax));
    const bool is_softmax_v1 = static_cast<bool>(ov::as_type_ptr<ov::op::v1::Softmax>(softmax));
    if (!is_softmax_v8 && !is_softmax_v1) {
        return false;
    }

    auto transpose_before = ov::as_type_ptr<ov::op::v1::Transpose>(softmax->input_value(0).get_node_shared_ptr());
    if (!transpose_before || transpose_before->get_input_size() != 2 || transpose_before->get_output_size() != 1 ||
        transpose_before->output(0).get_target_inputs().size() != 1) {
        return false;
    }

    const auto& input_pshape = transpose_before->get_input_partial_shape(0);
    if (input_pshape.is_dynamic() || !input_pshape.rank().is_static()) {
        return false;
    }
    const auto rank = static_cast<size_t>(input_pshape.rank().get_length());

    auto perm_before = get_constant_permutation(transpose_before->input_value(1));
    auto perm_after = get_constant_permutation(transpose_after->input_value(1));
    if (!perm_before || !perm_after || perm_before->size() != rank || perm_after->size() != rank) {
        return false;
    }
    if (!is_valid_permutation(*perm_before) || !is_valid_permutation(*perm_after)) {
        return false;
    }

    int64_t softmax_axis = -1;
    if (auto softmax_v8 = ov::as_type_ptr<ov::op::v8::Softmax>(softmax)) {
        softmax_axis = softmax_v8->get_axis();
    } else if (auto softmax_v1 = ov::as_type_ptr<ov::op::v1::Softmax>(softmax)) {
        softmax_axis = static_cast<int64_t>(softmax_v1->get_axis());
    } else {
        return false;
    }

    if (softmax_axis < 0) {
        softmax_axis += static_cast<int64_t>(rank);
    }
    if (softmax_axis < 0 || softmax_axis >= static_cast<int64_t>(rank)) {
        return false;
    }

    const auto new_axis = (*perm_before)[static_cast<size_t>(softmax_axis)];
    const auto composed_perm = compose_permutations(*perm_before, *perm_after);

    std::shared_ptr<ov::Node> new_softmax;
    if (is_softmax_v8) {
        new_softmax = std::make_shared<ov::op::v8::Softmax>(transpose_before->input_value(0), new_axis);
    } else {
        new_softmax = std::make_shared<ov::op::v1::Softmax>(transpose_before->input_value(0),
                                                            static_cast<size_t>(new_axis));
    }
    ov::copy_runtime_info({transpose_before, softmax}, new_softmax);

    if (is_identity_permutation_vector(composed_perm)) {
        new_softmax->set_friendly_name(transpose_after->get_friendly_name());
        ov::copy_runtime_info(ov::NodeVector{transpose_after}, new_softmax);
        ov::replace_node(transpose_after, new_softmax);
        return true;
    }

    auto composed_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{composed_perm.size()}, composed_perm);
    auto new_transpose = std::make_shared<ov::op::v1::Transpose>(new_softmax, composed_const);
    new_transpose->set_friendly_name(transpose_after->get_friendly_name());
    ov::copy_runtime_info(ov::NodeVector{transpose_after}, ov::NodeVector{composed_const, new_transpose});
    ov::replace_node(transpose_after, new_transpose);
    return true;
}

bool fold_dfl_softmax_expectation(const std::shared_ptr<ov::Node>& node) {
    auto final_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(node);
    if (!final_reshape || final_reshape->get_input_size() != 2 || final_reshape->get_output_size() != 1) {
        return false;
    }

    auto conv = ov::as_type_ptr<ov::op::v1::Convolution>(final_reshape->input_value(0).get_node_shared_ptr());
    if (!conv || conv->get_input_size() != 2 || conv->get_output_size() != 1 ||
        conv->output(0).get_target_inputs().size() != 1) {
        return false;
    }
    if (conv->get_strides() != ov::Strides{1, 1} || conv->get_dilations() != ov::Strides{1, 1} ||
        conv->get_pads_begin() != ov::CoordinateDiff{0, 0} || conv->get_pads_end() != ov::CoordinateDiff{0, 0}) {
        return false;
    }

    const auto weights_const = get_constant_from_source_local(conv->input_value(1));
    if (!weights_const) {
        return false;
    }
    const auto weights_shape = weights_const->get_shape();
    if (weights_shape.size() != 4 || weights_shape[0] != 1 || weights_shape[2] != 1 || weights_shape[3] != 1) {
        return false;
    }

    auto transpose_after = ov::as_type_ptr<ov::op::v1::Transpose>(conv->input_value(0).get_node_shared_ptr());
    if (!transpose_after || transpose_after->get_input_size() != 2 || transpose_after->get_output_size() != 1 ||
        transpose_after->output(0).get_target_inputs().size() != 1) {
        return false;
    }
    auto softmax_node = transpose_after->input_value(0).get_node_shared_ptr();
    auto softmax_v8 = ov::as_type_ptr<ov::op::v8::Softmax>(softmax_node);
    auto softmax_v1 = ov::as_type_ptr<ov::op::v1::Softmax>(softmax_node);
    if ((!softmax_v8 && !softmax_v1) || softmax_node->get_output_size() != 1 ||
        softmax_node->output(0).get_target_inputs().size() != 1) {
        return false;
    }
    auto transpose_before = ov::as_type_ptr<ov::op::v1::Transpose>(softmax_node->input_value(0).get_node_shared_ptr());
    if (!transpose_before || transpose_before->get_input_size() != 2 || transpose_before->get_output_size() != 1 ||
        transpose_before->output(0).get_target_inputs().size() != 1) {
        return false;
    }

    const auto perm_before = get_constant_permutation(transpose_before->input_value(1));
    const auto perm_after = get_constant_permutation(transpose_after->input_value(1));
    if (!perm_before || !perm_after) {
        return false;
    }
    int64_t softmax_axis = softmax_v8 ? softmax_v8->get_axis() : static_cast<int64_t>(softmax_v1->get_axis());
    if (*perm_before != std::vector<int64_t>({0, 3, 1, 2}) ||
        *perm_after != std::vector<int64_t>({0, 3, 2, 1}) ||
        softmax_axis != 3) {
        return false;
    }

    const auto& source_pshape = transpose_before->get_input_partial_shape(0);
    if (source_pshape.is_dynamic() || !source_pshape.rank().is_static() ||
        source_pshape.rank().get_length() != 4) {
        return false;
    }
    const auto source_shape = transpose_before->get_input_shape(0);
    if (source_shape.size() != 4 || source_shape[2] != weights_shape[1]) {
        return false;
    }

    auto new_softmax = std::make_shared<ov::op::v8::Softmax>(transpose_before->input_value(0), 2);

    const auto weights_values = weights_const->cast_vector<float>();
    auto flatten_shape = ov::op::v0::Constant::create(ov::element::i64,
                                                       ov::Shape{4},
                                                       std::vector<int64_t>{static_cast<int64_t>(source_shape[0]),
                                                                            static_cast<int64_t>(source_shape[1] *
                                                                                                 source_shape[2]),
                                                                            static_cast<int64_t>(1),
                                                                            static_cast<int64_t>(source_shape[3])});
    auto flatten = std::make_shared<ov::op::v1::Reshape>(new_softmax, flatten_shape, false);

    const size_t coord_count = static_cast<size_t>(source_shape[1]);
    const size_t bins_per_coord = weights_values.size();
    const size_t flattened_channels = coord_count * bins_per_coord;
    std::vector<float> sparse_weights_values(coord_count * flattened_channels, 0.0f);
    for (size_t coord = 0; coord < coord_count; ++coord) {
        const size_t channel_offset = coord * bins_per_coord;
        const size_t row_offset = coord * flattened_channels;
        for (size_t bin = 0; bin < bins_per_coord; ++bin) {
            sparse_weights_values[row_offset + channel_offset + bin] = weights_values[bin];
        }
    }
    ov::Shape sparse_weights_shape{coord_count, flattened_channels, 1, 1};
    auto sparse_weights = ov::op::v0::Constant::create(weights_const->get_element_type(),
                                                       sparse_weights_shape,
                                                       sparse_weights_values);
    auto sparse_conv = std::make_shared<ov::op::v1::Convolution>(flatten,
                                                                 sparse_weights,
                                                                 ov::Strides{1, 1},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::Strides{1, 1},
                                                                 ov::op::PadType::EXPLICIT);
    auto new_reshape =
        std::make_shared<ov::op::v1::Reshape>(sparse_conv, final_reshape->input_value(1), final_reshape->get_special_zero());

    new_reshape->set_friendly_name(final_reshape->get_friendly_name());
    ov::copy_runtime_info(ov::NodeVector{transpose_before, softmax_node, transpose_after, conv, final_reshape},
                          ov::NodeVector{new_softmax, flatten_shape, flatten, sparse_weights, sparse_conv, new_reshape});
    ov::replace_node(final_reshape, new_reshape);
    return true;
}

bool fold_transpose_unary_transpose(const std::shared_ptr<ov::Node>& node) {
    auto transpose_after = ov::as_type_ptr<ov::op::v1::Transpose>(node);
    if (!transpose_after || transpose_after->get_input_size() != 2 || transpose_after->get_output_size() != 1) {
        return false;
    }

    auto unary = ov::as_type_ptr<ov::op::util::UnaryElementwiseArithmetic>(
        transpose_after->input_value(0).get_node_shared_ptr());
    if (!unary || unary->get_input_size() != 1 || unary->get_output_size() != 1 ||
        unary->output(0).get_target_inputs().size() != 1) {
        return false;
    }

    auto transpose_before = ov::as_type_ptr<ov::op::v1::Transpose>(unary->input_value(0).get_node_shared_ptr());
    if (!transpose_before || transpose_before->get_input_size() != 2 || transpose_before->get_output_size() != 1 ||
        transpose_before->output(0).get_target_inputs().size() != 1) {
        return false;
    }

    const auto& input_pshape = transpose_before->get_input_partial_shape(0);
    if (input_pshape.is_dynamic() || !input_pshape.rank().is_static()) {
        return false;
    }
    const auto rank = static_cast<size_t>(input_pshape.rank().get_length());

    auto perm_before = get_constant_permutation(transpose_before->input_value(1));
    auto perm_after = get_constant_permutation(transpose_after->input_value(1));
    if (!perm_before || !perm_after || perm_before->size() != rank || perm_after->size() != rank) {
        return false;
    }
    if (!is_valid_permutation(*perm_before) || !is_valid_permutation(*perm_after)) {
        return false;
    }

    auto new_unary = unary->clone_with_new_inputs({transpose_before->input_value(0)});
    ov::copy_runtime_info(ov::NodeVector{transpose_before, unary}, new_unary);

    const auto composed_perm = compose_permutations(*perm_before, *perm_after);
    if (is_identity_permutation_vector(composed_perm)) {
        new_unary->set_friendly_name(transpose_after->get_friendly_name());
        ov::copy_runtime_info(ov::NodeVector{transpose_after}, new_unary);
        ov::replace_node(transpose_after, new_unary);
        return true;
    }

    auto composed_const = make_i64_constant(composed_perm);
    auto new_transpose = std::make_shared<ov::op::v1::Transpose>(new_unary, composed_const);
    new_transpose->set_friendly_name(transpose_after->get_friendly_name());
    ov::copy_runtime_info(ov::NodeVector{transpose_after}, ov::NodeVector{composed_const, new_transpose});
    ov::replace_node(transpose_after, new_transpose);
    return true;
}

bool deduplicate_transpose_reshape_branch(const std::shared_ptr<ov::Node>& node) {
    auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(node);
    if (!reshape || reshape->get_input_size() != 2 || reshape->get_output_size() != 1) {
        return false;
    }

    auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(reshape->input_value(0).get_node_shared_ptr());
    if (!transpose || transpose->get_input_size() != 2 || transpose->get_output_size() != 1 ||
        transpose->output(0).get_target_inputs().size() != 1) {
        return false;
    }

    const auto permutation = get_constant_permutation(transpose->input_value(1));
    const auto reshape_pattern = get_constant_i64_values(reshape->input_value(1));
    if (!permutation || !reshape_pattern) {
        return false;
    }

    auto source = transpose->input_value(0).get_node_shared_ptr();
    if (!source) {
        return false;
    }

    const size_t source_port = transpose->input_value(0).get_index();
    auto canonical = reshape;
    for (const auto& target_input : source->output(source_port).get_target_inputs()) {
        auto sibling_transpose = ov::as_type_ptr<ov::op::v1::Transpose>(target_input.get_node()->shared_from_this());
        if (!sibling_transpose || sibling_transpose == transpose ||
            sibling_transpose->get_input_size() != 2 || sibling_transpose->get_output_size() != 1 ||
            sibling_transpose->output(0).get_target_inputs().size() != 1) {
            continue;
        }
        const auto sibling_perm = get_constant_permutation(sibling_transpose->input_value(1));
        if (!sibling_perm || *sibling_perm != *permutation) {
            continue;
        }
        auto sibling_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(
            sibling_transpose->output(0).get_target_inputs().begin()->get_node()->shared_from_this());
        if (!sibling_reshape || sibling_reshape == reshape ||
            sibling_reshape->get_input_size() != 2 || sibling_reshape->get_output_size() != 1) {
            continue;
        }
        const auto sibling_pattern = get_constant_i64_values(sibling_reshape->input_value(1));
        if (!sibling_pattern || *sibling_pattern != *reshape_pattern) {
            continue;
        }
        if (sibling_reshape->get_special_zero() != reshape->get_special_zero()) {
            continue;
        }
        if (sibling_reshape->get_friendly_name() < canonical->get_friendly_name()) {
            canonical = sibling_reshape;
        }
    }

    if (canonical == reshape) {
        return false;
    }
    return ov::replace_output_update_name(reshape->output(0), canonical->output(0));
}

bool eliminate_noop_reshape(const std::shared_ptr<ov::Node>& node) {
    auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(node);
    if (!reshape || reshape->get_input_size() < 1 || reshape->get_output_size() != 1) {
        return false;
    }
    const auto& in_pshape = reshape->get_input_partial_shape(0);
    const auto& out_pshape = reshape->get_output_partial_shape(0);
    if (in_pshape.is_dynamic() || out_pshape.is_dynamic() || in_pshape != out_pshape) {
        return false;
    }
    return ov::replace_output_update_name(reshape->output(0), reshape->input_value(0));
}

bool eliminate_noop_squeeze_or_unsqueeze(const std::shared_ptr<ov::Node>& node) {
    const bool is_squeeze = ov::is_type<ov::op::v0::Squeeze>(node) || ov::is_type<ov::op::v15::Squeeze>(node);
    const bool is_unsqueeze = ov::is_type<ov::op::v0::Unsqueeze>(node);
    if ((!is_squeeze && !is_unsqueeze) || node->get_input_size() < 1 || node->get_output_size() != 1) {
        return false;
    }
    const auto& in_pshape = node->get_input_partial_shape(0);
    const auto& out_pshape = node->get_output_partial_shape(0);
    if (in_pshape.is_dynamic() || out_pshape.is_dynamic() || in_pshape != out_pshape) {
        return false;
    }
    return ov::replace_output_update_name(node->output(0), node->input_value(0));
}

bool eliminate_single_input_concat(const std::shared_ptr<ov::Node>& node) {
    auto concat = ov::as_type_ptr<ov::op::v0::Concat>(node);
    if (!concat || concat->get_input_size() != 1 || concat->get_output_size() != 1) {
        return false;
    }
    const auto& in_pshape = concat->get_input_partial_shape(0);
    const auto& out_pshape = concat->get_output_partial_shape(0);
    if (in_pshape.is_dynamic() || out_pshape.is_dynamic() || in_pshape != out_pshape) {
        return false;
    }
    return ov::replace_output_update_name(concat->output(0), concat->input_value(0));
}

template <typename PadT>
bool eliminate_noop_pad_impl(const std::shared_ptr<ov::Node>& node) {
    auto pad = ov::as_type_ptr<PadT>(node);
    if (!pad || pad->get_output_size() != 1 || pad->get_input_size() < 3) {
        return false;
    }
    const auto& in_pshape = pad->get_input_partial_shape(0);
    const auto& out_pshape = pad->get_output_partial_shape(0);
    if (in_pshape.is_dynamic() || out_pshape.is_dynamic() || in_pshape != out_pshape) {
        return false;
    }
    const auto& pads_begin = pad->get_pads_begin();
    const auto& pads_end = pad->get_pads_end();
    if (pads_begin.size() != pads_end.size()) {
        return false;
    }
    for (size_t i = 0; i < pads_begin.size(); ++i) {
        if (pads_begin[i] != 0 || pads_end[i] != 0) {
            return false;
        }
    }
    return ov::replace_output_update_name(pad->output(0), pad->input_value(0));
}

bool eliminate_noop_pad(const std::shared_ptr<ov::Node>& node) {
    return eliminate_noop_pad_impl<ov::op::v1::Pad>(node) || eliminate_noop_pad_impl<ov::op::v12::Pad>(node);
}

bool is_zero_constant_pad_value(const ov::Output<ov::Node>& pad_value) {
    auto constant = ov::as_type_ptr<const ov::op::v0::Constant>(pad_value.get_node_shared_ptr());
    if (!constant) {
        return false;
    }
    const ov::Tensor tensor = constant->get_tensor_view();
    if (tensor.get_size() == 0) {
        return false;
    }
    switch (tensor.get_element_type()) {
        case ov::element::f16: {
            const auto* data = tensor.data<const ov::float16>();
            return std::all_of(data, data + tensor.get_size(), [](ov::float16 v) { return static_cast<float>(v) == 0.0f; });
        }
        case ov::element::f32: {
            const auto* data = tensor.data<const float>();
            return std::all_of(data, data + tensor.get_size(), [](float v) { return v == 0.0f; });
        }
        case ov::element::i32: {
            const auto* data = tensor.data<const int32_t>();
            return std::all_of(data, data + tensor.get_size(), [](int32_t v) { return v == 0; });
        }
        case ov::element::i64: {
            const auto* data = tensor.data<const int64_t>();
            return std::all_of(data, data + tensor.get_size(), [](int64_t v) { return v == 0; });
        }
        case ov::element::u8: {
            const auto* data = tensor.data<const uint8_t>();
            return std::all_of(data, data + tensor.get_size(), [](uint8_t v) { return v == 0; });
        }
        default:
            return false;
    }
}

template <typename ConvT>
bool merge_zero_pad_into_conv_impl(const std::shared_ptr<ov::Node>& node) {
    auto conv = ov::as_type_ptr<ConvT>(node);
    if (!conv || conv->get_input_size() != 2 || conv->get_output_size() != 1) {
        return false;
    }
    if (conv->get_auto_pad() != ov::op::PadType::EXPLICIT) {
        return false;
    }
    auto pad = ov::as_type_ptr<ov::op::util::PadBase>(conv->input_value(0).get_node_shared_ptr());
    if (!pad || pad->get_input_size() < 3 || pad->get_output_size() != 1) {
        return false;
    }
    if (pad->output(0).get_target_inputs().size() != 1) {
        return false;
    }
    if (pad->get_pad_mode() != ov::op::PadMode::CONSTANT) {
        return false;
    }
    if (pad->get_input_partial_shape(0).is_dynamic() || pad->get_output_partial_shape(0).is_dynamic()) {
        return false;
    }
    const auto& in_shape = pad->get_input_shape(0);
    const auto& out_shape = pad->get_output_shape(0);
    if (in_shape.size() != 4 || out_shape.size() != 4) {
        return false;
    }
    if (pad->get_input_size() > 3 && !is_zero_constant_pad_value(pad->input_value(3))) {
        return false;
    }
    const auto& pads_begin = pad->get_pads_begin();
    const auto& pads_end = pad->get_pads_end();
    if (pads_begin.size() != 4 || pads_end.size() != 4) {
        return false;
    }
    if (pads_begin[0] != 0 || pads_begin[1] != 0 || pads_end[0] != 0 || pads_end[1] != 0) {
        return false;
    }
    if (pads_begin[2] < 0 || pads_begin[3] < 0 || pads_end[2] < 0 || pads_end[3] < 0) {
        return false;
    }

    ov::CoordinateDiff new_pads_begin = conv->get_pads_begin();
    ov::CoordinateDiff new_pads_end = conv->get_pads_end();
    if (new_pads_begin.size() != 2 || new_pads_end.size() != 2) {
        return false;
    }
    new_pads_begin[0] += pads_begin[2];
    new_pads_begin[1] += pads_begin[3];
    new_pads_end[0] += pads_end[2];
    new_pads_end[1] += pads_end[3];

    auto new_conv = std::make_shared<ConvT>(pad->input_value(0),
                                            conv->input_value(1),
                                            conv->get_strides(),
                                            new_pads_begin,
                                            new_pads_end,
                                            conv->get_dilations(),
                                            ov::op::PadType::EXPLICIT);
    new_conv->set_friendly_name(conv->get_friendly_name());
    ov::copy_runtime_info({pad, conv}, new_conv);
    ov::replace_node(conv, new_conv);
    return true;
}

bool merge_zero_pad_into_conv(const std::shared_ptr<ov::Node>& node) {
    return merge_zero_pad_into_conv_impl<ov::op::v1::Convolution>(node) ||
           merge_zero_pad_into_conv_impl<ov::op::v1::GroupConvolution>(node);
}

}  // namespace

bool GfxLayoutCleanup::run_on_model(const std::shared_ptr<ov::Model>& model) {
    OPENVINO_ASSERT(model, "GfxLayoutCleanup: model is null");

    bool changed = false;
    bool local_change = false;
    do {
        local_change = false;
        const auto ordered_ops = model->get_ordered_ops();
        for (const auto& node : ordered_ops) {
            if (ov::as_type_ptr<ov::op::v0::Parameter>(node) ||
                ov::as_type_ptr<ov::op::v0::Constant>(node) ||
                ov::as_type_ptr<ov::op::v0::Result>(node)) {
                continue;
            }
            if (fold_dfl_softmax_expectation(node)) {
                local_change = true;
                changed = true;
            }
        }
        if (local_change) {
            model->validate_nodes_and_infer_types();
            continue;
        }
        for (const auto& node : ordered_ops) {
            if (ov::as_type_ptr<ov::op::v0::Parameter>(node) ||
                ov::as_type_ptr<ov::op::v0::Constant>(node) ||
                ov::as_type_ptr<ov::op::v0::Result>(node)) {
                continue;
            }
            if (fold_transpose_softmax_transpose(node) ||
                fold_transpose_unary_transpose(node) ||
                deduplicate_transpose_reshape_branch(node) ||
                eliminate_identity_transpose(node) ||
                eliminate_noop_reshape(node) ||
                eliminate_noop_squeeze_or_unsqueeze(node) || eliminate_single_input_concat(node) ||
                eliminate_noop_pad(node) || merge_zero_pad_into_conv(node)) {
                local_change = true;
                changed = true;
            }
        }
        if (local_change) {
            model->validate_nodes_and_infer_types();
        }
    } while (local_change);

    return changed;
}

}  // namespace transforms
}  // namespace gfx_plugin
}  // namespace ov
