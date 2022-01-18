// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_transpose_arm.hpp"
#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertTranspose, "ConvertTranspose", 0);
ArmPlugin::pass::ConvertTranspose::ConvertTranspose() {
    auto transpose = ngraph::pattern::wrap_type<opset::Transpose>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto transpose = std::dynamic_pointer_cast<opset::Transpose>(m.get_match_root());
        if (!transpose) {
            return false;
        }

        if (transpose->get_shape().size() > 4) {
            return false;
        }

        auto arm_transpose = std::make_shared<opset::ArmTranspose>(transpose->input_value(0), transpose->input_value(1));
        arm_transpose->set_friendly_name(transpose->get_friendly_name());
        ngraph::copy_runtime_info(transpose, arm_transpose);
        ngraph::replace_node(transpose, arm_transpose);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(transpose, "ConvertTranspose");
    register_matcher(m, callback);
}
