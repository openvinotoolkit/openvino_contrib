// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cpp/ie_cnn_network.h>
#include <gtest/gtest.h>

#include <cuda_test_constants.hpp>
#include <fstream>
#include <ie_ngraph_utils.hpp>
#include <map>
#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/variant.hpp>
#include <queue>
#include <sstream>
#include <string>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "ngraph_conversion_tests/conv_bias_fusion.hpp"
#include "transformations/rt_info/primitives_priority_attribute.hpp"

using namespace NGraphConversionTestsDefinitions;

namespace {

INSTANTIATE_TEST_CASE_P(smoke_Basic, ConvBiasFusion,
                        ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                        ConvBiasFusion::getTestCaseName);

}  // namespace
