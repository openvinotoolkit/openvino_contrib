// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/finalize_trailing_nodes.hpp"

#include <numeric>

#include "opset/opset.hpp"
#include "transformations/utils/utils.hpp"
#include <ngraph/rt_info.hpp>

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::FinalizeTrailingNodes, "FinalizeTrailingNodes", 0);
bool ArmPlugin::pass::FinalizeTrailingNodes::run_on_function(std::shared_ptr<ov::Model> m) {
    // Adding Result node for trailing nodes
    bool is_modified = false;
    for (const auto& node : m->get_ops()) {
        for (const auto& out : node->outputs()) {
            auto inputs = out.get_target_inputs();
            if (inputs.empty() && !std::dynamic_pointer_cast<opset::Result>(node)) {
                auto result = std::make_shared<opset::Result>(out);
                result->set_friendly_name(ngraph::op::util::get_ie_output_name(out));
                m->add_results({result});
                is_modified = true;
            }
        }
    }
    return is_modified;
}
