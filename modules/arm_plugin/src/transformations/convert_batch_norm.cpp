// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_batch_norm.hpp"

#include <numeric>

#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "opset/opset.hpp"

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertBatchNormInference, "ConvertBatchNormInference", 0);
ArmPlugin::pass::ConvertBatchNormInference::ConvertBatchNormInference() {
    auto batch_norm = ngraph::pattern::wrap_type<opset::BatchNormInference>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        enum Input {Features, Gamma, Beta, Mean, Variance};
        auto node = std::dynamic_pointer_cast<opset::BatchNormInference>(m.get_match_root());

        if (!node) {
            return false;
        }

        auto inp_shape = node->get_shape();
        if (inp_shape.size() == 4) {
            return false;
        }

        std::vector<int64_t> new_shape(inp_shape.begin(), inp_shape.end());
        if (inp_shape.size() > 4) {
            for (size_t i = 4; i < inp_shape.size(); i++) {
                new_shape[3] *= inp_shape[i];
            }
        }
        new_shape.resize(4, 1);

        auto shape = std::make_shared<opset::Constant>(ngraph::element::i64, ngraph::Shape{4}, new_shape);

        auto reshape = std::make_shared<opset::Reshape>(node->input_value(Input::Features), shape, true);
        auto bn      = std::make_shared<opset::BatchNormInference>(reshape,
                                                                   node->input_value(Input::Gamma),
                                                                   node->input_value(Input::Beta),
                                                                   node->input_value(Input::Mean),
                                                                   node->input_value(Input::Variance),
                                                                   node->get_eps_value());

        shape = std::make_shared<opset::Constant>(ngraph::element::i64, ngraph::Shape{inp_shape.size()},
                                                  std::vector<int64_t>(inp_shape.begin(), inp_shape.end()));
        auto reshape_back = std::make_shared<opset::Reshape>(bn, shape, true);


        reshape_back->set_friendly_name(node->get_friendly_name());
        ngraph::copy_runtime_info(node, {reshape, bn, reshape_back});
        ngraph::replace_node(node, reshape_back);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(batch_norm, "ConvertBatchNormInference");
    register_matcher(m, callback);
}
