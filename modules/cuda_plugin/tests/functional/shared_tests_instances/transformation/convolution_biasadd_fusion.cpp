// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_test_constants.hpp>
#include <ngraph_conversion_tests/conv_bias_fusion.hpp>

using namespace NGraphConversionTestsDefinitions;

namespace {

// TODO: enable whenever Conv2DBiasAdd Op implementation available
INSTANTIATE_TEST_CASE_P(DISABLED_smoke_Basic, ConvBiasFusion,
                        ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                        ConvBiasFusion::getTestCaseName);

}  // namespace
