// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "convert_batch_to_space_to_arm.hpp"
#include "ngraph/rt_info.hpp"
#include "opset/opset.hpp"
#include <openvino/pass/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertBatchToSpaceToARM, "ConvertBatchToSpaceToARM", 0);
ArmPlugin::pass::ConvertBatchToSpaceToARM::ConvertBatchToSpaceToARM() {
    auto batch_to_space = ov::pass::pattern::wrap_type<ov::op::v1::BatchToSpace>();
    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto batch_to_space = std::dynamic_pointer_cast<ov::op::v1::BatchToSpace>(m.get_match_root());
        if (!batch_to_space) {
            return false;
        }

        enum BatchToSpace {Data, BlockShape, CropsBegin, CropsEnd};
        auto batch_to_space_arm = std::make_shared<opset::v1::ArmBatchToSpace>(
                                        batch_to_space->input_value(BatchToSpace::Data),
                                        batch_to_space->input_value(BatchToSpace::BlockShape),
                                        batch_to_space->input_value(BatchToSpace::CropsBegin),
                                        batch_to_space->input_value(BatchToSpace::CropsEnd));

        batch_to_space_arm->set_friendly_name(batch_to_space->get_friendly_name() + "/arm");
        ov::copy_runtime_info(batch_to_space, batch_to_space_arm);
        ov::replace_node(batch_to_space, batch_to_space_arm);
        return true;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(batch_to_space, "ConvertBatchToSpaceToARM");
    register_matcher(m, callback);
}

