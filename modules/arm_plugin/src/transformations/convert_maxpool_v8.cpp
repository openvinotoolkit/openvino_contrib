// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_maxpool_v8.hpp"

#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertMaxPoolV8, "ConvertMaxPoolV8", 0);

ArmPlugin::pass::ConvertMaxPoolV8::ConvertMaxPoolV8() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<ngraph::op::v8::MaxPool>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                                 ngraph::pattern::has_static_shape()), "ConvertMaxPoolV8");

    register_matcher(m, [&](ngraph::pattern::Matcher& m) {
        auto maxpool = std::dynamic_pointer_cast<ngraph::op::v8::MaxPool>(m.get_match_root());
        if (!maxpool) {
            return false;
        }

        // Pooling indices only supported for kernel size 2x2
        if ((maxpool->get_input_shape(0).size() != 4) ||
            (maxpool->get_kernel() != ngraph::Shape{2, 2}) ||
            (maxpool->get_dilations() != ngraph::Strides{1, 1}) ||
            (maxpool->get_axis() != 0)) {
            return false;
        }
        auto new_maxpool = std::make_shared<ngraph::op::v8::MaxPool>(maxpool->input_value(0),
                                                                     maxpool->get_strides(),
                                                                     maxpool->get_dilations(),
                                                                     maxpool->get_pads_begin(),
                                                                     maxpool->get_pads_end(),
                                                                     maxpool->get_kernel(),
                                                                     maxpool->get_rounding_type(),
                                                                     maxpool->get_auto_pad(),
                                                                     ngraph::element::u32,
                                                                     maxpool->get_axis());
        ngraph::Output<ngraph::Node> output_0 = new_maxpool->output(0);
        ngraph::Output<ngraph::Node> output_1_convert = new_maxpool->output(1);

        output_1_convert = std::make_shared<opset::Convert>(output_1_convert,
                                                            maxpool->get_output_element_type(1));
        output_1_convert.get_node_shared_ptr()->set_friendly_name(maxpool->get_friendly_name() + "/convert.1");

        new_maxpool->set_friendly_name(maxpool->get_friendly_name());
        ngraph::copy_runtime_info(maxpool, new_maxpool);
        ngraph::replace_node(maxpool, {output_0, output_1_convert});
        return true;
    });
}