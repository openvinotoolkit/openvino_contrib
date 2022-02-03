// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "opset/opset.hpp"
#include "store_result_name.hpp"
#include "transformations/utils/utils.hpp"

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::StoreResultName, "StoreResultName", 0);
bool ArmPlugin::pass::StoreResultName::run_on_function(std::shared_ptr<ov::Model> m) {
    for (auto&& node : m->get_results()) {
        IE_ASSERT(node->inputs().size() == 1);
        auto input = node->input(0);
        auto sourceOutput = input.get_source_output();
        const auto outputName = ngraph::op::util::get_ie_output_name(sourceOutput);
        node->get_rt_info().emplace("ResultName", outputName);
    }
    return false;
}
