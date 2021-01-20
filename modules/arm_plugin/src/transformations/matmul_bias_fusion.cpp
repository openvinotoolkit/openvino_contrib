// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "matmul_bias_fusion.hpp"

#include <memory>

#include <ngraph/rt_info.hpp>
#include "opset/opset.hpp"

using namespace ArmPlugin;

ArmPlugin::pass::MatMulBiasFusion::MatMulBiasFusion() : GraphRewrite() {
    auto input0 = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::i64, ngraph::Shape{1, 1});
    auto input1 = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::i64, ngraph::Shape{1, 1});
    auto matmul = std::make_shared<opset::MatMul>(input0, input1);
    auto add    = std::make_shared<opset::Add>(matmul, std::make_shared<ngraph::pattern::op::Label>());

    ngraph::graph_rewrite_callback callback = [](ngraph::pattern::Matcher& m) {
        auto add    = m.get_match_root();
        enum Inputs {Data, Weights};
        auto matmul = std::dynamic_pointer_cast<opset::MatMul>(add->input(0).get_source_output().get_node_shared_ptr());
        auto bias   = std::dynamic_pointer_cast<opset::Constant>(add->input(1).get_source_output().get_node_shared_ptr());

        if (!matmul) {
            matmul = std::dynamic_pointer_cast<opset::MatMul>(add->input(1).get_source_output().get_node_shared_ptr());
            bias   = std::dynamic_pointer_cast<opset::Constant>(add->input(0).get_source_output().get_node_shared_ptr());
        }

        if (!matmul || !bias) {
            return false;
        }

        auto input_a = matmul->input(Inputs::Data).get_source_output().get_node_shared_ptr();
        auto input_b = matmul->input(Inputs::Weights).get_source_output().get_node_shared_ptr();
        if (matmul->get_shape().size() != 2 || !std::dynamic_pointer_cast<opset::Constant>(input_b)) {
            return false;
        }

        ngraph::NodeVector new_ops;
        if (matmul->get_transpose_a()) {
            input_a = std::make_shared<opset::Transpose>(matmul->input(0).get_source_output().get_node_shared_ptr(),
                                                                opset::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1, 0}));
            input_a->set_friendly_name(matmul->get_friendly_name() + "/transpose_a");
            new_ops.push_back(input_a);
        }
        auto matmul_bias = std::make_shared<opset::MatMulBias>(input_a, input_b, bias, matmul->get_transpose_b());
        new_ops.push_back(matmul_bias);

        matmul_bias->set_friendly_name(matmul->get_friendly_name());
        ngraph::copy_runtime_info(matmul, new_ops);
        ngraph::replace_node(matmul, matmul_bias);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(add, "MatMulBiasFusion");
    this->add_matcher(m, callback, ngraph::pass::PassProperty::CHANGE_DYNAMIC_STATE);
}
