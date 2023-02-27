// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "opset/opset.hpp"
#include "store_result_name.hpp"
#include "transformations/utils/utils.hpp"

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::StoreResultName, "StoreResultName");
bool ArmPlugin::pass::StoreResultName::run_on_model(const std::shared_ptr<ov::Model>& m) {
    for (auto&& node : m->get_results()) {
        IE_ASSERT(node->inputs().size() == 1);
        auto input = node->input(0);
        auto sourceOutput = input.get_source_output();
        const auto outputName = ov::op::util::get_ie_output_name(sourceOutput);
        node->get_rt_info().emplace("ResultName", outputName);
    }
    return false;
}
