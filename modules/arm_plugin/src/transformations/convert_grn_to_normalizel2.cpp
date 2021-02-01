// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_grn_to_normalizel2.hpp"

#include <numeric>

#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>

ArmPlugin::pass::ConvertGRN::ConvertGRN() {
    auto grn = std::make_shared<opset::GRN>(ngraph::pattern::any_input(), 1.f);

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto grn = std::dynamic_pointer_cast<opset::GRN>(m.get_match_root());
        if (!grn) {
            return false;
        }
        float eps = grn->get_bias();
        auto axes = opset::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto norm = std::make_shared<opset::NormalizeL2>(grn->input_value(0), axes, eps, ngraph::op::EpsMode::ADD);

        norm->set_friendly_name(grn->get_friendly_name());
        ngraph::copy_runtime_info(grn, norm);
        ngraph::replace_node(grn, norm);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(grn, "ConvertGRN");
    register_matcher(m, callback);
}
