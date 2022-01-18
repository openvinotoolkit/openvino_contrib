// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_strided_slice_arm.hpp"
#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertStridedSliceToArm, "ConvertStridedSliceToArm", 0);
ArmPlugin::pass::ConvertStridedSliceToArm::ConvertStridedSliceToArm() {
    auto slice = ngraph::pattern::wrap_type<opset::StridedSlice>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto slice = std::dynamic_pointer_cast<opset::StridedSlice>(m.get_match_root());
        if (!slice) {
            return false;
        }

        if (slice->get_input_shape(0).size() > 4) {
            return false;
        }

        auto&& begin_node  = std::dynamic_pointer_cast<ngraph::op::Constant>(slice->input_value(1).get_node_shared_ptr());
        auto&& end_node    = std::dynamic_pointer_cast<ngraph::op::Constant>(slice->input_value(2).get_node_shared_ptr());
        auto&& stride_node = std::dynamic_pointer_cast<ngraph::op::Constant>(slice->input_value(3).get_node_shared_ptr());
        if (!begin_node || !end_node || !stride_node) {
            return false;
        }

        auto ellipsisMask    = slice->get_ellipsis_mask();
        auto newAxis         = slice->get_new_axis_mask();
        auto shrinkAxis      = slice->get_shrink_axis_mask();
        bool addOrReduceDims = std::find(newAxis.begin(), newAxis.end(), 1) != newAxis.end() ||
                               std::find(shrinkAxis.begin(), shrinkAxis.end(), 1) != shrinkAxis.end();
        if (addOrReduceDims && std::find(ellipsisMask.begin(), ellipsisMask.end(), 1) != ellipsisMask.end()) {
            return false;
        }

        auto arm_slice = std::make_shared<opset::ArmStridedSlice>(slice->input_value(0),
                                                                  slice->input_value(1),
                                                                  slice->input_value(2),
                                                                  slice->input_value(3),
                                                                  slice->get_begin_mask(),
                                                                  slice->get_end_mask(),
                                                                  slice->get_new_axis_mask(),
                                                                  slice->get_shrink_axis_mask(),
                                                                  slice->get_ellipsis_mask());
        arm_slice->set_friendly_name(slice->get_friendly_name());
        ngraph::copy_runtime_info(slice, arm_slice);
        ngraph::replace_node(slice, arm_slice);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(slice, "ConvertStridedSliceToArm");
    register_matcher(m, callback);
}
