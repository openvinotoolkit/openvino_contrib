// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "convert_depth_to_space_to_arm.hpp"
#include "ngraph/rt_info.hpp"
#include "opset/opset.hpp"
#include <openvino/pass/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertDepthToSpaceToARM, "ConvertDepthToSpaceToARM", 0);
ArmPlugin::pass::ConvertDepthToSpaceToARM::ConvertDepthToSpaceToARM() {
    auto depth_to_space = ov::pass::pattern::wrap_type<ov::op::v0::DepthToSpace>();
    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto depth_to_space = std::dynamic_pointer_cast<ov::op::v0::DepthToSpace>(m.get_match_root());
        if (!depth_to_space) {
            return false;
        }

        auto depth_to_space_arm = std::make_shared<opset::v0::ArmDepthToSpace>(
                                                        depth_to_space->input_value(0),
                                                        depth_to_space->get_mode(),
                                                        depth_to_space->get_block_size());

        depth_to_space_arm->set_friendly_name(depth_to_space->get_friendly_name() + "/arm");
        ov::copy_runtime_info(depth_to_space, depth_to_space_arm);
        ov::replace_node(depth_to_space, depth_to_space_arm);
        return true;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(depth_to_space, "ConvertDepthToSpaceToARM");
    register_matcher(m, callback);
}

