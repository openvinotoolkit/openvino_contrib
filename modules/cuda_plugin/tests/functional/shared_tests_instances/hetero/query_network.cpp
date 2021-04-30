// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <cuda_test_constants.hpp>

#include "hetero/query_network.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

namespace {
using namespace HeteroTests;

auto ConvBias = ngraph::builder::subgraph::makeConvBias();

INSTANTIATE_TEST_CASE_P(smoke_FullySupportedTopologies, QueryNetworkTest,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_CUDA, "HETERO:CUDA", "MULTI:CUDA"),
                                ::testing::Values(ConvBias)),
                        QueryNetworkTest::getTestCaseName);
}  // namespace
