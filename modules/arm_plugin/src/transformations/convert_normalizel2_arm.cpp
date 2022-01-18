// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_normalizel2_arm.hpp"
#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertNormalizeL2ToArm, "ConvertNormalizeL2ToArm", 0);
ArmPlugin::pass::ConvertNormalizeL2ToArm::ConvertNormalizeL2ToArm() {
    auto norml2 = ngraph::pattern::wrap_type<opset::NormalizeL2>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto norml2 = std::dynamic_pointer_cast<opset::NormalizeL2>(m.get_match_root());
        if (!norml2) {
            return false;
        }

        if (norml2->get_eps_mode() == ngraph::op::EpsMode::ADD) {
            return false;
        }

        auto&& axes = norml2->get_reduction_axes().to_vector();
        int axis = norml2->get_input_shape(0).size() - axes[0] - 1;
        if (axes.size() != 1 || axis > 2) {
            return false;
        }

        auto arm_norml2 = std::make_shared<opset::ArmNormalizeL2>(norml2->input_value(0),
                                                                  norml2->input_value(1),
                                                                  norml2->get_eps(),
                                                                  norml2->get_eps_mode());
        arm_norml2->set_friendly_name(norml2->get_friendly_name());
        ngraph::copy_runtime_info(norml2, arm_norml2);
        ngraph::replace_node(norml2, arm_norml2);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(norml2, "ConvertNormalizeL2ToArm");
    register_matcher(m, callback);
}
