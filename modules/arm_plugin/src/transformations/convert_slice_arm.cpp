// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_slice_arm.hpp"
#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "ngraph/validation_util.hpp"

ArmPlugin::pass::ConvertSliceToArm::ConvertSliceToArm() {
    auto slice = ngraph::pattern::wrap_type<ngraph::op::v8::Slice>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto slice = std::dynamic_pointer_cast<ngraph::op::v8::Slice>(m.get_match_root());
        if (!slice) {
            return false;
        }

        if (slice->get_input_shape(0).size() > 4) {
            return false;
        }

        auto&& begin_node  = std::dynamic_pointer_cast<ngraph::op::Constant>(slice->input_value(1).get_node_shared_ptr());
        auto&& end_node    = std::dynamic_pointer_cast<ngraph::op::Constant>(slice->input_value(2).get_node_shared_ptr());
        auto&& step_node = std::dynamic_pointer_cast<ngraph::op::Constant>(slice->input_value(3).get_node_shared_ptr());
        if (!begin_node || !end_node || !step_node) {
            return false;
        }

        const auto axes_const  = slice->get_input_size() > 4 ?
            ngraph::get_constant_from_source(slice->input_value(4)) :
            slice->get_default_const_axes(slice->input_value(1));

        auto arm_slice = std::make_shared<opset::ArmSlice>(slice->input_value(0),
                                                           slice->input_value(1),
                                                           slice->input_value(2),
                                                           slice->input_value(3),
                                                           axes_const);

        arm_slice->set_friendly_name(slice->get_friendly_name());
        ngraph::copy_runtime_info(slice, arm_slice);
        ngraph::replace_node(slice, arm_slice);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(slice, "ConvertSliceToArm");
    register_matcher(m, callback);
}
