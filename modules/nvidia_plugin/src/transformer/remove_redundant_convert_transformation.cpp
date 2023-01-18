// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remove_redundant_convert_transformation.hpp"

#include <gsl/gsl_assert>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <openvino/op/convert.hpp>

namespace ngraph::pass {

NGRAPH_RTTI_DEFINITION(ngraph::pass::RemoveRedundantConvertTransformation, "RemoveRedundantConvertTransformation", 0);

namespace {

bool remove_redundant_convert(ngraph::pattern::Matcher& m) {
    auto node = std::dynamic_pointer_cast<ov::op::v0::Convert>(m.get_match_root());
    Expects(node);

    auto in_element_type = node->get_input_element_type(0);
    auto out_element_type = node->get_output_element_type(0);
    if (in_element_type == out_element_type) {
        return ov::replace_output_update_name(node->output(0), node->input_value(0));
    }

    const auto& inputs = node->output(0).get_target_inputs();
    if (inputs.size() == 1) {
        const auto& next_in = *inputs.begin();
        if (dynamic_cast<ov::op::v0::Convert*>(next_in.get_node())) {
            return ov::replace_output_update_name(node->output(0), node->input_value(0));
        }
    }

    return false;
}

}  // namespace

RemoveRedundantConvertTransformation::RemoveRedundantConvertTransformation() {
    const auto op = ngraph::pattern::wrap_type<ov::op::v0::Convert>();
    const auto m = std::make_shared<ngraph::pattern::Matcher>(op, "RemoveRedundantConvertTransformation");

    matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) { return remove_redundant_convert(m); };

    register_matcher(m, callback);
}

}  // namespace ngraph::pass
