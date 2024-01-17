// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/cc/pass/itt.hpp"
#include "fuse_matmul_add.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include <openvino/op/add.hpp>
#include <openvino/op/matmul.hpp>
#include <ops/matmul.hpp>

using namespace ov::pass::pattern;

namespace {
std::pair<std::shared_ptr<ov::op::v0::MatMul>, std::shared_ptr<ov::op::v0::Constant>> get_matmul_constant_nodes(const std::shared_ptr<ov::Node>& add_node) {
    if (std::dynamic_pointer_cast<ov::op::v0::Constant>(add_node->get_input_node_shared_ptr(1))) {
        return {std::dynamic_pointer_cast<ov::op::v0::MatMul>(add_node->get_input_node_shared_ptr(0)),
                std::dynamic_pointer_cast<ov::op::v0::Constant>(add_node->get_input_node_shared_ptr(1))};
    } else if (std::dynamic_pointer_cast<ov::op::v0::Constant>(add_node->get_input_node_shared_ptr(0))) {
        return {std::dynamic_pointer_cast<ov::op::v0::MatMul>(add_node->get_input_node_shared_ptr(1)),
                std::dynamic_pointer_cast<ov::op::v0::Constant>(add_node->get_input_node_shared_ptr(0))};
    }
    return {nullptr, nullptr};
}

bool is_add_to_be_fused(const ov::Output<ov::Node>& output) {
    auto add_node = std::dynamic_pointer_cast<ov::op::v1::Add>(output.get_node_shared_ptr());
    if (!add_node || add_node->is_dynamic()) {
        return false;
    }
    std::shared_ptr<ov::op::v0::MatMul> matmul_node;
    std::shared_ptr<ov::op::v0::Constant> constant_node;
    std::tie(matmul_node, constant_node) = get_matmul_constant_nodes(add_node);
    if (!matmul_node || !constant_node || matmul_node->is_dynamic()) {
        return false;
    }

    auto matrix_A_shape = matmul_node->get_input_shape(0);
    auto matrix_B_shape = matmul_node->get_input_shape(1);
    const auto matrix_shape = matmul_node->get_output_shape(0);
    ov::nvidia_gpu::MatMulOp::BroadcastToMatrix(matrix_A_shape);
    ov::nvidia_gpu::MatMulOp::BroadcastToMatrix(matrix_B_shape);
    const auto matmul_batch = std::max(ov::nvidia_gpu::MatMulOp::GetMatrixNumBatches(matrix_A_shape),
                                       ov::nvidia_gpu::MatMulOp::GetMatrixNumBatches(matrix_B_shape));

    auto const_shape = constant_node->get_output_shape(0);
    ov::nvidia_gpu::MatMulOp::BroadcastToMatrix(const_shape);
    const auto const_batch = ov::nvidia_gpu::MatMulOp::GetMatrixNumBatches(const_shape);
    const auto const_shape_size = ov::shape_size(const_shape);
    const auto matrix_shape_size = ov::shape_size(matrix_shape);
    const auto num_auto_const_batch = matrix_shape_size / const_shape_size;
    const auto matmul_shape_dividable = matrix_shape_size % const_shape_size;
    if (matmul_batch < const_batch || matmul_shape_dividable != 0 || num_auto_const_batch > 1) {
        return false;
    }
    return true;
}
} // namespace

namespace ov::nvidia_gpu::pass {
bool fuse_matmul_and_add(Matcher &m) {
    // Decompose Divide into Multiply with Power operations
    auto add_node = std::dynamic_pointer_cast<ov::op::v1::Add>(m.get_match_root());
    auto consumers = add_node->output(0).get_target_inputs();
    std::shared_ptr<ov::op::v0::MatMul> matmul_node;
    std::shared_ptr<ov::op::v0::Constant> constant_node;
    std::tie(matmul_node, constant_node) = get_matmul_constant_nodes(add_node);
    const auto fully_connected_node =
        std::make_shared<ov::nvidia_gpu::nodes::FullyConnected>(matmul_node->get_input_source_output(0),
                                                                matmul_node->get_input_source_output(1),
                                                                constant_node,
                                                                matmul_node->get_transpose_a(),
                                                                matmul_node->get_transpose_b());
    fully_connected_node->set_friendly_name(add_node->get_friendly_name());
    ov::copy_runtime_info({matmul_node, add_node}, fully_connected_node);

    for (auto input : consumers) {
        input.replace_source_output(fully_connected_node);
    }
    return true;
}

FullyConnectedTransformation::FullyConnectedTransformation() {
    MATCHER_SCOPE(FullyConnectedTransformation);
    auto matmul = wrap_type<ov::op::v0::MatMul>(consumers_count(1));
    auto bias = wrap_type<ov::op::v0::Constant>();
    auto add0 = wrap_type<ov::op::v1::Add>({matmul, bias}, is_add_to_be_fused);
    auto add1 = wrap_type<ov::op::v1::Add>({bias, matmul}, is_add_to_be_fused);
    auto result = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{add0, add1});

    matcher_pass_callback callback = [](Matcher &m) { return fuse_matmul_and_add(m); };

    auto m = std::make_shared<Matcher>(result, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::nvidia_gpu::pass
