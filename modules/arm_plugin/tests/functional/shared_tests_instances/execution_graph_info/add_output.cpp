// Copyright (C) 2020-2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_constants.hpp>
#include "execution_graph_tests/add_output.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

InferenceEngine::CNNNetwork getTargetNetwork() {
    return InferenceEngine::CNNNetwork { ngraph::builder::subgraph::makeConvPoolRelu() };
}

std::vector<addOutputsParams> testCases = {
    addOutputsParams(getTargetNetwork(), {"Pool_1"}, CommonTestUtils::DEVICE_CPU)
};

INSTANTIATE_TEST_CASE_P(AddOutputBasic, AddOutputsTest,
                        ::testing::ValuesIn(testCases),
                        AddOutputsTest::getTestCaseName);
