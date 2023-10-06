// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "cuda_test_constants.hpp"
#include "single_op_tests/transpose.hpp"


namespace {

using namespace ov::test;
using namespace ov::test::utils;
using ov::test::TransposeLayerTest;

const std::vector<ov::element::Type> model_types = {
    ov::element::f16,
    ov::element::f32,
};

const std::vector<std::vector<ov::Shape>> input_shapes = {
    {{256, 3, 100, 100}},
    {{1, 2048, 1, 1}},
};

const std::vector<std::vector<size_t>> input_order = {
    std::vector<size_t>{0, 3, 2, 1},
    // Empty inputs are currently unsupported in nvidia_gpu.
    //        std::vector<size_t>{},
};

const auto params = testing::Combine(testing::ValuesIn(input_order),
                                     testing::ValuesIn(model_types),
                                     testing::ValuesIn(static_shapes_to_test_representation(input_shapes)),
                                     testing::Values(DEVICE_NVIDIA));

INSTANTIATE_TEST_CASE_P(smoke_Transpose,
                        TransposeLayerTest,
                        params,
                        TransposeLayerTest::getTestCaseName);

}  // namespace
