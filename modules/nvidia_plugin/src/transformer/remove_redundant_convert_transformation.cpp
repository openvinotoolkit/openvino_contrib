// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remove_redundant_convert_transformation.hpp"

#include <openvino/core/except.hpp>
#include <openvino/op/convert.hpp>

#include "openvino/cc/ngraph/itt.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov::pass::pattern;

namespace ov::nvidia_gpu::pass {
namespace {

bool remove_redundant_convert(Matcher& m) {
    auto node = std::dynamic_pointer_cast<ov::op::v0::Convert>(m.get_match_root());
    OPENVINO_ASSERT(node);

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
    MATCHER_SCOPE(RemoveRedundantConvertTransformation);
    const auto op = ov::pass::pattern::wrap_type<ov::op::v0::Convert>();
    const auto m = std::make_shared<ov::pass::pattern::Matcher>(op, matcher_name);

    matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) { return remove_redundant_convert(m); };

    register_matcher(m, callback);
}

}  // namespace ov::nvidia_gpu::pass
