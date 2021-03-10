// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_pool1d_to_pool2d.hpp"

#include <numeric>

#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

template <class Pool>
ngraph::matcher_pass_callback ArmPlugin::pass::ConvertPool1DBase::convert_pool1d_to_pool2d() {
    return [&](ngraph::pattern::Matcher& m) {
        auto pool = std::dynamic_pointer_cast<Pool>(m.get_match_root());
        if (!pool) {
            return false;
        }

        auto input_shape = pool->get_input_shape(0);
        // is Pool1D
        if (input_shape.size() != 3) {
            return false;
        }

        auto input   = pool->input_value(0);
        auto input2d_shape = input_shape;
        input2d_shape.push_back(1);
        auto in2d_shape = std::make_shared<opset::Constant>(ngraph::element::i64, ngraph::Shape{4}, input2d_shape);

        auto input2d   = std::make_shared<opset::Reshape>(input, in2d_shape, true);
        std::shared_ptr<ngraph::Node> pool2d;
        if (std::is_same<Pool, opset::AvgPool>()) {
            auto avg_pool = std::dynamic_pointer_cast<opset::AvgPool>(pool);
            pool2d = std::make_shared<opset::AvgPool>(input2d,
                                             ngraph::Strides{avg_pool->get_strides()[0], 1},
                                             ngraph::Shape{avg_pool->get_pads_begin()[0], 0},
                                             ngraph::Shape{avg_pool->get_pads_end()[0], 0},
                                             ngraph::Shape{avg_pool->get_kernel()[0], 1},
                                             avg_pool->get_exclude_pad(),
                                             avg_pool->get_rounding_type(),
                                             avg_pool->get_auto_pad());
        } else {
            auto max_pool = std::dynamic_pointer_cast<opset::MaxPool>(pool);
            pool2d = std::make_shared<opset::MaxPool>(input2d,
                                             ngraph::Strides{max_pool->get_strides()[0], 1},
                                             ngraph::Shape{max_pool->get_pads_begin()[0], 0},
                                             ngraph::Shape{max_pool->get_pads_end()[0], 0},
                                             ngraph::Shape{max_pool->get_kernel()[0], 1},
                                             max_pool->get_rounding_type(),
                                             max_pool->get_auto_pad());
        }

        auto in_shape = std::make_shared<opset::Constant>(ngraph::element::i64, ngraph::Shape{3}, pool->get_output_shape(0));
        auto reshape = std::make_shared<opset::Reshape>(pool2d, in_shape, true);

        reshape->set_friendly_name(pool->get_friendly_name());
        ngraph::copy_runtime_info(pool, {input2d, pool2d, reshape});
        ngraph::replace_node(pool, reshape);
        return true;
    };
}

ArmPlugin::pass::ConvertMaxPool1D::ConvertMaxPool1D() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
        ngraph::pattern::wrap_type<opset::MaxPool>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                    ngraph::pattern::has_static_shape()), "ConvertMaxPooling1D");
    register_matcher(m, convert_pool1d_to_pool2d<opset::MaxPool>());
}

ArmPlugin::pass::ConvertAvgPool1D::ConvertAvgPool1D() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<opset::AvgPool>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                        ngraph::pattern::has_static_shape()), "ConvertAvgPooling1D");
    register_matcher(m, convert_pool1d_to_pool2d<opset::AvgPool>());
}