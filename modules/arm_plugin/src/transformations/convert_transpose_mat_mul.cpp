// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_transpose_mat_mul.hpp"

#include <numeric>
#include <cmath>
#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <openvino/core/validation_util.hpp>

using namespace ov;

enum MatMulInput {MatA, MatB};


std::shared_ptr<ArmPlugin::opset::Constant> get_transpose_order(const std::shared_ptr<const Node>& node) {
    switch (node->get_shape().size()) {
        case 2:
            return ArmPlugin::opset::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
        case 3:
            return ArmPlugin::opset::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
        case 4:
            return ArmPlugin::opset::Constant::create(ov::element::i64, ov::Shape{3}, {0, 1, 3, 2});
    }
    return {};
}

std::shared_ptr<ArmPlugin::opset::Constant> get_reshape_order(const std::shared_ptr<const Node>& node_a,
                                                              const std::shared_ptr<const Node>& node_b,
                                                              bool transpose) {
    auto max_reshape_size = std::max(node_a->get_shape().size(), node_b->get_shape().size());
    auto min_reshape_size = std::max(node_a->get_shape().size(), node_b->get_shape().size());
    if (node_a->get_shape().size() < 3) {
        std::vector<size_t> shape_order_2d = {1, node_a->get_shape().at(0)};
        std::vector<size_t> shape_order_3d = {1, 1, node_a->get_shape().at(0)};
        std::vector<size_t> shape_order_4d = {1, 1, 1, node_a->get_shape().at(0)};
        switch (max_reshape_size) {
            case 1:
            case 2:
                if (!transpose) {
                    return ArmPlugin::opset::Constant::create(ov::element::i64, ov::Shape{2}, shape_order_2d);
                } else {
                    std::reverse(shape_order_2d.begin(), shape_order_2d.end());
                    return ArmPlugin::opset::Constant::create(ov::element::i64, ov::Shape{2}, shape_order_2d);
                }
            case 3:
                return ArmPlugin::opset::Constant::create(ov::element::i64, ov::Shape{3}, shape_order_3d);
            case 4:
                return ArmPlugin::opset::Constant::create(ov::element::i64, ov::Shape{3}, shape_order_3d);
        }
    } else if (node_a->get_shape().size() == 3) {
        std::vector<size_t> shape_order_1d_to_3d = {1, 1, node_a->get_shape().at(0)};
        std::vector<size_t> shape_order_2d_to_3d = {1, node_a->get_shape().at(0), node_a->get_shape().at(1)};
        switch (min_reshape_size) {
            case 1:
                return ArmPlugin::opset::Constant::create(ov::element::i64, ov::Shape{2}, shape_order_1d_to_3d);
            case 2:
                return ArmPlugin::opset::Constant::create(ov::element::i64, ov::Shape{2}, shape_order_2d_to_3d);
        }
    } else if (node_a->get_shape().size() == 4) {
        std::vector<size_t> shape_order_1d_to_4d = {1, 1, 1, node_a->get_shape().at(0)};
        std::vector<size_t> shape_order_2d_to_4d = {1, 1, node_a->get_shape().at(0), node_a->get_shape().at(1)};
        std::vector<size_t> shape_order_3d_to_4d = {1, node_a->get_shape().at(0), node_a->get_shape().at(1), node_a->get_shape().at(2)};
        switch (min_reshape_size) {
            case 1:
                return ArmPlugin::opset::Constant::create(ov::element::i64, ov::Shape{2}, shape_order_1d_to_4d);
            case 2:
                return ArmPlugin::opset::Constant::create(ov::element::i64, ov::Shape{2}, shape_order_2d_to_4d);
            case 3:
                return ArmPlugin::opset::Constant::create(ov::element::i64, ov::Shape{2}, shape_order_3d_to_4d);
        }
    }
    return {};
}

OPENVINO_OP(ArmPlugin::pass::ConvertTransposeMatMul, "ConvertTransposeMatMul");
ArmPlugin::pass::ConvertTransposeMatMul::ConvertTransposeMatMul() {
    auto matmul = std::make_shared<opset::MatMul>(ngraph::pattern::any_input(), ngraph::pattern::any_input());
    matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto matmul = std::dynamic_pointer_cast<opset::MatMul>(m.get_match_root());
        if (!matmul) {
            return false;
        }

        auto input_a = matmul->input_value(MatMulInput::MatA).get_node_shared_ptr();
        auto input_b = matmul->input_value(MatMulInput::MatB).get_node_shared_ptr();
        auto shape_a = matmul->input(MatMulInput::MatA).get_shape();
        auto shape_b = matmul->input(MatMulInput::MatB).get_shape();
        auto transpose_a = matmul->get_transpose_a();
        auto transpose_b = matmul->get_transpose_b();

        if (shape_a.size() == 4 || shape_b.size() == 4) {
            return false;
        }

        ov::NodeVector new_ops;
        if (shape_a.size() == 1 && shape_b.size() == 1) {
            // Reshape A
            input_a = std::make_shared<opset::Reshape>(input_a, get_reshape_order(input_a, input_b, false), true);
            input_a->set_friendly_name(matmul->get_friendly_name() + "/reshape_a");
            new_ops.push_back(input_a);
            // Reshape B
            input_b = std::make_shared<opset::Reshape>(input_b, get_reshape_order(input_b, input_a, true), true);
            input_b->set_friendly_name(matmul->get_friendly_name() + "/reshape_b");
            new_ops.push_back(input_b);
            transpose_a = transpose_b = false;
        } else if (shape_a.size() == 1 && shape_b.size() > 1) {
            // Reshape A
            input_a = std::make_shared<opset::Reshape>(input_a, get_reshape_order(input_a, input_b, false), true);
            input_a->set_friendly_name(matmul->get_friendly_name() + "/reshape_a");
            new_ops.push_back(input_a);
        } else if (shape_a.size() > 1 && shape_b.size() == 1) {
            // Reshape B
            input_b = std::make_shared<opset::Reshape>(input_b, get_reshape_order(input_b, input_a, true), true);
            input_b->set_friendly_name(matmul->get_friendly_name() + "/reshape_b");
            new_ops.push_back(input_b);
        }
        if (transpose_a) {
            // Transpose A
            input_a = std::make_shared<opset::ArmTranspose>(input_a, get_transpose_order(input_a));
            input_a->set_friendly_name(matmul->get_friendly_name() + "/transpose_a");
            new_ops.push_back(input_a);
            transpose_a = false;
        }
        if (transpose_b) {
            // Transpose B
            input_b = std::make_shared<opset::ArmTranspose>(input_b, get_transpose_order(input_b));
            input_b->set_friendly_name(matmul->get_friendly_name() + "/transpose_b");
            new_ops.push_back(input_b);
            transpose_b = false;
        }
        auto new_matmul = std::make_shared<opset::MatMul>(input_a, input_b, transpose_a, transpose_b);
        new_ops.push_back(new_matmul);
        new_matmul->set_friendly_name(matmul->get_friendly_name() + "/new");
        ngraph::copy_runtime_info(matmul, new_ops);
        ngraph::replace_node(matmul, new_matmul);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matmul, "ConvertTransposeMatMul");
    this->register_matcher(m, callback);
}