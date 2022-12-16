// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_gathernd_arm.hpp"
#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertGatherNDV5toV8, "ConvertGatherNDV5toV8", 0);
ArmPlugin::pass::ConvertGatherNDV5toV8::ConvertGatherNDV5toV8() {
    auto gather_nd = ngraph::pattern::wrap_type<ov::op::v5::GatherND>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto gather_nd = std::dynamic_pointer_cast<ov::op::v5::GatherND>(m.get_match_root());
        if (!gather_nd) {
            return false;
        }

        auto gather_nd_v8 = std::make_shared<ov::op::v8::GatherND>(gather_nd->input_value(0),
                                                                   gather_nd->input_value(1),
                                                                   gather_nd->get_batch_dims());
        gather_nd_v8->set_friendly_name(gather_nd->get_friendly_name());
        ngraph::copy_runtime_info(gather_nd, gather_nd_v8);
        ngraph::replace_node(gather_nd, gather_nd_v8);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(gather_nd, "ConvertGatherNDV5toV8");
    register_matcher(m, callback);
}
