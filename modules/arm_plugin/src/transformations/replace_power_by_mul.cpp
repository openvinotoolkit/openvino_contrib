// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_ceiling.hpp"

#include <numeric>

#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

ArmPlugin::pass::ReplacePowerByMul::ReplacePowerByMul() {
    auto constant_pattern = ngraph::pattern::wrap_type<opset::Constant>();
    auto power_pattern = ngraph::pattern::wrap_type<opset::Power>({
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
        constant});

    register_matcher(std::make_shared<ngraph::pattern::Matcher>(power, "ReplacePowerByMul"), [=] (ngraph::pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_value_map();
        auto power = std::dynamic_pointer_cast<opset::Power>(pattern_map[power_pattern]);
        if (!power) {
            return false;
        }
        auto constant = std::dynamic_pointer_cast<opset::constant>(pattern_map[constant_pattern]);
        if (!constant) {
            return false;
        }
        if (ov::shape_size(constant->get_shape()) != 1) {
            return false;
        }
        auto p = constant->cast_vector<float>()[0];

        if (p == 1) {
            return ov::replace_output_update_name(power->output(0), power->input_value(0));
        } else if (p == 2) {
            auto mul = std::make_shared<opset::Multiply>(power->input(0), power->input(0));
            mul->set_friendly_name(power->get_friendly_name());
            ngraph::copy_runtime_info({power, constant}, mul);
            ngraph::replace_node(power, mul);
            return true;
        }
        return false;
    });
}
