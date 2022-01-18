// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_mvn_arm.hpp"
#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertMVN, "ConvertMVN", 0);
ArmPlugin::pass::ConvertMVN::ConvertMVN() {
    auto mvn = ngraph::pattern::wrap_type<opset::MVN>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto mvn = std::dynamic_pointer_cast<opset::MVN>(m.get_match_root());
        if (!mvn) {
            return false;
        }

        if (mvn->get_shape().size() != 2) {
            return false;
        }

        if (mvn->get_eps_mode() == ngraph::op::MVNEpsMode::INSIDE_SQRT || !mvn->get_normalize_variance()) {
            return false;
        }

        auto&& axes = std::dynamic_pointer_cast<ngraph::op::Constant>(mvn->input_value(1).get_node_shared_ptr());
        if (!axes || axes->cast_vector<int64_t>()[0] == 1) {
            return false;
        }

        auto arm_mvn = std::make_shared<opset::ArmMVN>(mvn->input_value(0), mvn->get_eps());
        arm_mvn->set_friendly_name(mvn->get_friendly_name());
        ngraph::copy_runtime_info(mvn, arm_mvn);
        ngraph::replace_node(mvn, arm_mvn);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mvn, "ConvertMVN");
    register_matcher(m, callback);
}
