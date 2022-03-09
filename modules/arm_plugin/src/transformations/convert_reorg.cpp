// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_reorg.hpp"

#include <numeric>

#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertReorgYolo, "ConvertReorgYolo", 0);
ArmPlugin::pass::ConvertReorgYolo::ConvertReorgYolo() {
    auto reorg = ngraph::pattern::wrap_type<opset::ReorgYolo>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto reorg = std::dynamic_pointer_cast<opset::ReorgYolo>(m.get_match_root());
        if (!reorg) {
            return false;
        }
        auto strides = reorg->get_strides();
        if (strides[0] == 1 && strides[1] == 1) {
            return false;
        }

        auto stride = strides[0];
        auto input_shape = reorg->get_input_shape(0);

        auto axis = opset::Constant::create<int64_t>(ngraph::element::i64, ngraph::Shape{}, {0});
        auto shape  = opset::Constant::create<int64_t>(ngraph::element::i64, ngraph::Shape{4},
            { static_cast<long>(input_shape[1] * input_shape[2] / (stride * stride)),
              static_cast<long>(stride),
              static_cast<long>(input_shape[3]),
              static_cast<long>(stride) });
        auto order  = opset::Constant::create<int64_t>(ngraph::element::i64, ngraph::Shape{4}, {1, 3, 0, 2});

        auto split = std::make_shared<opset::Split>(reorg->input_value(0), axis, input_shape[0]);
        ngraph::NodeVector slices;
        for (auto&& slice : split->outputs()) {
            auto reshape = std::make_shared<opset::Reshape>(slice, shape, true);
            auto tr_forward = std::make_shared<opset::Transpose>(reshape, order);
            slices.push_back(tr_forward);
        }

        auto concat = std::make_shared<opset::Concat>(slices, 0);
        auto out_shape = std::make_shared<opset::Constant>(ngraph::element::i64, ngraph::Shape{4}, reorg->get_output_shape(0));
        auto reshape_back = std::make_shared<opset::Reshape>(concat, out_shape, true);

        reshape_back->set_friendly_name(reorg->get_friendly_name());
        ngraph::copy_runtime_info(reorg, {split, concat, reshape_back});
        ngraph::replace_node(reorg, reshape_back);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(reorg, "ConvertReorgYolo");
    register_matcher(m, callback);
}
