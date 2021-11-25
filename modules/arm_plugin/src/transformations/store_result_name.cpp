// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "opset/opset.hpp"
#include "store_result_name.hpp"
#include "transformations/utils/utils.hpp"

bool ArmPlugin::pass::StoreResultName::run_on_function(std::shared_ptr<ngraph::Function> f) {
    for (auto&& node : f->get_results()) {
        IE_ASSERT(node->inputs().size() == 1);
        auto input = node->input(0);
        auto sourceOutput = input.get_source_output();
        const auto outputName = ngraph::op::util::get_ie_output_name(sourceOutput);
        node->get_rt_info().emplace("ResultName", std::make_shared<ngraph::VariantWrapper<std::string>>(outputName));
    }
    return false;
}
