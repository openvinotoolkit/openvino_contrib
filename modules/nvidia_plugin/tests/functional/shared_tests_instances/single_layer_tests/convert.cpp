// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "single_op_tests/conversion.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include <cuda_test_constants.hpp>
#include <vector>

namespace {

using namespace ov::test;
using namespace ov::test::utils;

class ConversionCUDALayerTest : public ConversionLayerTest {};

TEST_P(ConversionCUDALayerTest, Inference) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

const std::vector<ConversionTypes> conversion_op_types = {
    ConversionTypes::CONVERT,
};

const std::vector<ov::Shape> in_shapes = {{1, 2, 3, 4}};

// List of precisions natively supported by CUDA.
// CUDA device supports only u8, f16 and f32 output precision
const std::vector<ov::element::Type> out_precisions = {
    ov::element::u8,
    ov::element::i16,
    ov::element::f16,
    ov::element::bf16,
    ov::element::f32,
};

// Supported formats are: boolean, f32, f6, i16 and u8
const std::vector<ov::element::Type> in_precisions = {
    ov::element::boolean,
    ov::element::u8,
    ov::element::i16,
    ov::element::f16,
    ov::element::bf16,
    ov::element::f32,
};

INSTANTIATE_TEST_SUITE_P(smoke_ConversionLayerTest,
                         ConversionCUDALayerTest,
                         ::testing::Combine(::testing::ValuesIn(conversion_op_types),
                                            ::testing::Values(static_shapes_to_test_representation(in_shapes)),
                                            ::testing::ValuesIn(in_precisions),
                                            ::testing::ValuesIn(out_precisions),
                                            ::testing::Values(DEVICE_NVIDIA)),
                         ConversionLayerTest::getTestCaseName);
}  // namespace
