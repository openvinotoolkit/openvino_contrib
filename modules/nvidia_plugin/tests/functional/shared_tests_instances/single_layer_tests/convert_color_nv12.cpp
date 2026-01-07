// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/convert_color_nv12.hpp"
#include "convert_color_common.hpp"

#include <cuda_test_constants.hpp>
#include <vector>

namespace {

using namespace ov::test;
using namespace ov::test::nvidia_gpu;
using namespace ov::test::utils;

class ConvertColorNV12CUDALayerTest : public ConvertColorCUDALayerTest, public ConvertColorNV12LayerTest {};

TEST_P(ConvertColorNV12CUDALayerTest, Inference) { run(); }

const std::vector<ov::Shape> in_shapes_nhwc = {
    {1, 10, 10, 1},
    {1, 50, 10, 1},
    {1, 100, 10, 1},
    {2, 10, 10, 1},
    {2, 50, 10, 1},
    {2, 100, 10, 1},
    {5, 10, 10, 1},
    {5, 50, 10, 1},
    {5, 100, 10, 1},
    {1, 96, 16, 1},
};

auto generate_input_static_shapes = [] (const std::vector<ov::Shape>& original_shapes, bool single_plane) {
    std::vector<std::vector<ov::Shape>> result_shapes;
    for (const auto& original_shape : original_shapes) {
        std::vector<ov::Shape> one_result_shapes;
        if (single_plane) {
            auto shape = original_shape;
            shape[1] = shape[1] * 3 / 2;
            one_result_shapes.push_back(shape);
        } else {
            auto shape = original_shape;
            one_result_shapes.push_back(shape);
            auto uvShape = ov::Shape{shape[0], shape[1] / 2, shape[2] / 2, 2};
            one_result_shapes.push_back(uvShape);
        }
        result_shapes.push_back(one_result_shapes);
    }
    return result_shapes;
};

auto in_shapes_single_plain_static     = generate_input_static_shapes(in_shapes_nhwc, true);
auto in_shapes_not_single_plain_static = generate_input_static_shapes(in_shapes_nhwc, false);

const std::vector<ov::element::Type> in_types = {
    ov::element::u8,
    ov::element::f32,
    ov::element::f16,
};

const auto test_case_values_single_plain = ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(in_shapes_single_plain_static)),
                                                              ::testing::ValuesIn(in_types),
                                                              ::testing::Bool(),
                                                              ::testing::Values(true),
                                                              ::testing::Values(DEVICE_NVIDIA));

INSTANTIATE_TEST_SUITE_P(smoke_TestsConvertColorNV12_single_plain,
                         ConvertColorNV12CUDALayerTest,
                         test_case_values_single_plain,
                         ConvertColorNV12CUDALayerTest::getTestCaseName);

const auto test_case_values_not_single_plain = ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(in_shapes_not_single_plain_static)),
                                                              ::testing::ValuesIn(in_types),
                                                              ::testing::Bool(),
                                                              ::testing::Values(false),
                                                              ::testing::Values(DEVICE_NVIDIA));

INSTANTIATE_TEST_SUITE_P(smoke_TestsConvertColorNV12_not_single_plain,
                         ConvertColorNV12CUDALayerTest,
                         test_case_values_not_single_plain,
                         ConvertColorNV12CUDALayerTest::getTestCaseName);


const auto test_case_accuracy_values = ::testing::Combine(::testing::Values(static_shapes_to_test_representation({ov::Shape{1, 96, 16, 1}})),
                                                          ::testing::Values(ov::element::u8),
                                                          ::testing::Values(false),
                                                          ::testing::Values(true),
                                                          ::testing::Values(DEVICE_NVIDIA));

INSTANTIATE_TEST_SUITE_P(smoke_TestsConvertColorNV12_acc,
                         ConvertColorNV12CUDALayerTest,
                         test_case_accuracy_values,
                         ConvertColorNV12LayerTest::getTestCaseName);

const auto test_case_accuracy_values_nightly = ::testing::Combine(::testing::Values(static_shapes_to_test_representation({ov::Shape{1, 65538, 256, 1}})),
                                                                  ::testing::Values(ov::element::u8),
                                                                  ::testing::Values(false),
                                                                  ::testing::Values(true),
                                                                  ::testing::Values(DEVICE_NVIDIA));

INSTANTIATE_TEST_SUITE_P(nightly_TestsConvertColorNV12_acc,
                         ConvertColorNV12CUDALayerTest,
                         test_case_accuracy_values_nightly,
                         ConvertColorNV12LayerTest::getTestCaseName);

}  // namespace
