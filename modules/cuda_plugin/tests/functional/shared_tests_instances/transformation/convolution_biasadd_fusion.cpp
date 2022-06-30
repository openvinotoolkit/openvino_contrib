// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_test_constants.hpp>

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <queue>
#include <map>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/variant.hpp>
#include <transformations/utils/utils.hpp>
#include <cpp/ie_cnn_network.h>
#include <ie_ngraph_utils.hpp>
#include "transformations/rt_info/primitives_priority_attribute.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph_conversion_tests/conv_bias_fusion.hpp"

using namespace NGraphConversionTestsDefinitions;

namespace {

INSTANTIATE_TEST_CASE_P(smoke_Basic, ConvBiasFusion,
                        ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                        ConvBiasFusion::getTestCaseName);

}  // namespace
