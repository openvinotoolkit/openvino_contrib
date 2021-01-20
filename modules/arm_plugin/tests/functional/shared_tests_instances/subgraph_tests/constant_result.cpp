// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/constant_result.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
    INSTANTIATE_TEST_CASE_P(Check, ConstantResultSubgraphTest, ::testing::Values("ARM"), ConstantResultSubgraphTest::getTestCaseName);
}  // namespace