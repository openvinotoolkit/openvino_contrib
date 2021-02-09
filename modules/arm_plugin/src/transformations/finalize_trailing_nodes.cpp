// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/finalize_trailing_nodes.hpp"

#include <numeric>

#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>


bool ArmPlugin::pass::FinalizeTrailingNodes::run_on_function(std::shared_ptr<ngraph::Function> f) {
    // Adding Result node for trailing nodes
    bool is_modified = false;
    for (const auto& node : f->get_ops()) {
        for (auto out : node->outputs()) {
            auto inputs = out.get_target_inputs();
            if (inputs.empty() && !std::dynamic_pointer_cast<opset::Result>(node)) {
                auto result = std::make_shared<opset::Result>(out);
                result->set_friendly_name(node->get_friendly_name() + "." +  std::to_string(out.get_index()));
                f->add_results({result});
                is_modified = true;
            }
        }
    }
    return is_modified;
}