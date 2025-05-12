// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "detection_output_fix_input_types_transformation.hpp"

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/detection_output.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov::pass::pattern;

namespace ov::nvidia_gpu::pass {

bool detection_output_fix_input_types(Matcher& m) {
    const auto d_out = std::dynamic_pointer_cast<ov::op::v0::DetectionOutput>(m.get_match_root());
    if (!d_out) {
        return false;
    }
    const auto& type = d_out->get_element_type();
    ov::OutputVector inputs;
    bool needs_transform = false;
    for (std::size_t i = 0; i < d_out->inputs().size(); ++i) {
        const auto& input_type = d_out->input(i).get_element_type();
        if (input_type != type) {
            needs_transform = true;
            const auto convert = std::make_shared<ov::op::v0::Convert>(d_out->input_value(i), type);
            ov::copy_runtime_info(d_out, convert);
            inputs.emplace_back(convert);
        } else {
            inputs.emplace_back(d_out->input_value(i));
        }
    }
    if (!needs_transform) {
        return false;
    }
    auto new_d_out = d_out->clone_with_new_inputs(inputs);
    new_d_out->set_friendly_name(d_out->get_friendly_name());
    ov::copy_runtime_info(d_out, new_d_out);
    ov::replace_node(d_out, new_d_out);
    return true;
}

DetectionOutputFixInputTypesTransformation::DetectionOutputFixInputTypesTransformation() {
    MATCHER_SCOPE(DetectionOutputFixInputTypesTransformation);

    const auto d_out = wrap_type<ov::op::v0::DetectionOutput>();
    matcher_pass_callback callback = [](Matcher& m) { return detection_output_fix_input_types(m); };

    const auto m = std::make_shared<Matcher>(d_out, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::nvidia_gpu::pass
