// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_mat_mul.hpp"

#include <numeric>

#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertMatMulToFC, "ConvertMatMulToFC", 0);
ArmPlugin::pass::ConvertMatMulToFC::ConvertMatMulToFC() {
    auto matmul = ngraph::pattern::wrap_type<opset::MatMul>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto matmul = std::dynamic_pointer_cast<opset::MatMul>(m.get_match_root());
        if (!matmul) {
            return false;
        }

        auto input_a = matmul->input_value(0);
        auto input_b = matmul->input_value(1);
        auto shape_a = input_a.get_shape();
        auto shape_b = input_b.get_shape();
        auto output_shape = matmul->get_shape();
        auto out_name = matmul->get_friendly_name();

        if (shape_a.size() <= 2 && shape_b.size() <= 2 &&
           !matmul->get_transpose_a() && matmul->get_transpose_b()) {
            return false;
        }

        auto create_transpose = [](ngraph::Output<ngraph::Node> node, const std::string& transpose_name) -> ngraph::Output<ngraph::Node> {
            std::vector<int64_t> transpose_order(node.get_shape().size());
            std::iota(transpose_order.begin(), transpose_order.end(), 0);
            std::swap(*(transpose_order.end() - 1), *(transpose_order.end() - 2));

            auto transpose = std::make_shared<opset::Transpose>(
                    node, opset::Constant::create(ngraph::element::i64, ngraph::Shape{transpose_order.size()}, transpose_order));
            transpose->set_friendly_name(transpose_name);
            return transpose->output(0);
        };

        ngraph::NodeVector new_ops;
        auto dst_shape = std::make_shared<opset::Constant>(ngraph::element::i64, ngraph::Shape{output_shape.size()},
                            std::vector<int64_t>(output_shape.begin(), output_shape.end()));

        size_t num_splits_a = ngraph::shape_size(ngraph::Shape(shape_a.begin(), shape_a.end() - 2));
        size_t num_splits_b = ngraph::shape_size(ngraph::Shape(shape_b.begin(), shape_b.end() - 2));

        auto input_shape = opset::Constant::create<int64_t>(ngraph::element::i64, ngraph::Shape{2},
            {-1ll, static_cast<long>(shape_a.back())});
        auto reshape_a = std::make_shared<opset::Reshape>(input_a, input_shape, true);
        new_ops.push_back(reshape_a);

        auto weight_shape = opset::Constant::create<int64_t>(ngraph::element::i64, ngraph::Shape{2},
            { static_cast<long>(num_splits_b * shape_b[shape_b.size() - 2]),
              static_cast<long>(shape_b.back()) });
        auto reshape_b = std::make_shared<opset::Reshape>(input_b, weight_shape, true);
        new_ops.push_back(reshape_b);

        auto axis = opset::Constant::create<int64_t>(ngraph::element::i64, ngraph::Shape{}, {0});
        auto split_a = std::make_shared<opset::Split>(reshape_a, axis, num_splits_a);
        auto split_b = std::make_shared<opset::Split>(reshape_b, axis, num_splits_b);

        ngraph::NodeVector matmul_sclices;
        for (size_t i = 0; i < std::max(num_splits_a, num_splits_b); i++) {
            size_t input_id  = i % num_splits_a;
            size_t weight_id = i % num_splits_b;
            // create copy of layer to avoid problem with memory manager
            ngraph::Output<ngraph::Node> first  = std::make_shared<opset::Concat>(ngraph::OutputVector{split_a->output(input_id)}, 0);
            ngraph::Output<ngraph::Node> second = std::make_shared<opset::Concat>(ngraph::OutputVector{split_b->output(weight_id)}, 0);

            if (matmul->get_transpose_a()) {
                first = create_transpose(first, out_name + "/tr_a_" + std::to_string(i));
                new_ops.push_back(first.get_node_shared_ptr());
            }
            if (!matmul->get_transpose_b()) {
                second = create_transpose(second, out_name + "/tr_b_" + std::to_string(i));
                new_ops.push_back(second.get_node_shared_ptr());
            }
            auto mat_mul = std::make_shared<opset::MatMul>(first, second, false, true);
            new_ops.push_back(mat_mul);
            matmul_sclices.push_back(mat_mul);
        }

        auto concat = std::make_shared<opset::Concat>(matmul_sclices, 0);
        auto reshape = std::make_shared<opset::Reshape>(concat, dst_shape, true);

        new_ops.push_back(concat);
        new_ops.push_back(reshape);

        reshape->set_friendly_name(out_name);
        ngraph::copy_runtime_info(matmul, new_ops);
        ngraph::replace_node(matmul, reshape);
        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(matmul, "ConvertMatMulToFC");
    register_matcher(m, callback);
}