// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_common.h>

#include <chrono>
#include <common_test_utils/common_utils.hpp>
#include <cuda_test_constants.hpp>
#include <ie_precision.hpp>
#include <map>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <string>
#include <vector>

#include "benchmark.hpp"
#include "cuda_eltwise.hpp"

namespace LayerTestsDefinitions {

namespace {

// Common parameters
const std::vector<ngraph::helpers::InputLayerType> input_layer_types = {ngraph::helpers::InputLayerType::CONSTANT,
                                                                        ngraph::helpers::InputLayerType::PARAMETER};

const ov::AnyMap additional_config = {};

// Smoke parameters
const std::vector<std::vector<ov::Shape>> smoke_shapes = {{{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}},
                                                          {{2, 3, 4, 5}, {2, 3, 4, 1}},
                                                          {{2, 3, 4, 5}, {2, 1, 1, 5}},
                                                          {{3, 1, 6, 1}, {1, 1, 1, 1}},
                                                          {{10, 10}, {10, 10}},
                                                          {{10, 10}, {1, 10}},
                                                          {{10, 10}, {1}},
                                                          {{8, 1, 6, 1}, {7, 1, 5}}};

const std::vector<CommonTestUtils::OpType> smoke_op_types = {CommonTestUtils::OpType::SCALAR,
                                                             CommonTestUtils::OpType::VECTOR};

const std::vector<ov::test::ElementType> add_precisions = {ov::test::ElementType::f16,
                                                           ov::test::ElementType::f32,
                                                           ov::test::ElementType::i32,
                                                           ov::test::ElementType::i16,
                                                           ov::test::ElementType::u8};

INSTANTIATE_TEST_CASE_P(
    smoke_Add,
    CudaEltwiseLayerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(smoke_shapes)),
                           ::testing::Values(ngraph::helpers::EltwiseTypes::ADD),
                           ::testing::ValuesIn(input_layer_types),
                           ::testing::ValuesIn(smoke_op_types),
                           ::testing::ValuesIn(add_precisions),
                           ::testing::Values(ov::test::ElementType::undefined),
                           ::testing::Values(ov::test::ElementType::undefined),
                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                           ::testing::Values(additional_config)),
        ::testing::Values(OperationMode::NORMAL)),
    CudaEltwiseLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    smoke_Add_U32,
    CudaEltwiseLayerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(smoke_shapes)),
                           ::testing::Values(ngraph::helpers::EltwiseTypes::ADD),
                           ::testing::ValuesIn(input_layer_types),
                           ::testing::ValuesIn(smoke_op_types),
                           ::testing::Values(ov::test::ElementType::u32),
                           ::testing::Values(ov::test::ElementType::u32),
                           ::testing::Values(ov::test::ElementType::u32),
                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                           ::testing::Values(additional_config)),
        ::testing::Values(OperationMode::NORMAL)),
    CudaEltwiseLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    smoke_Add_I64,
    CudaEltwiseLayerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(smoke_shapes)),
                           ::testing::Values(ngraph::helpers::EltwiseTypes::ADD),
                           ::testing::ValuesIn(input_layer_types),
                           ::testing::ValuesIn(smoke_op_types),
                           ::testing::Values(ov::test::ElementType::i64),
                           ::testing::Values(ov::test::ElementType::i64),
                           ::testing::Values(ov::test::ElementType::i64),
                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                           ::testing::Values(additional_config)),
        ::testing::Values(OperationMode::NORMAL)),
    CudaEltwiseLayerTest::getTestCaseName);

const std::vector<ov::test::ElementType> mul_precisions = {
    ov::test::ElementType::f16, ov::test::ElementType::f32, ov::test::ElementType::i32};

INSTANTIATE_TEST_CASE_P(
    smoke_Multiply,
    CudaEltwiseLayerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(smoke_shapes)),
                           ::testing::Values(ngraph::helpers::EltwiseTypes::MULTIPLY),
                           ::testing::ValuesIn(input_layer_types),
                           ::testing::ValuesIn(smoke_op_types),
                           ::testing::ValuesIn(mul_precisions),
                           ::testing::Values(ov::test::ElementType::undefined),
                           ::testing::Values(ov::test::ElementType::undefined),
                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                           ::testing::Values(additional_config)),
        ::testing::Values(OperationMode::NORMAL)),
    CudaEltwiseLayerTest::getTestCaseName);

const std::vector<ov::test::ElementType> sub_precisions = {
    ov::test::ElementType::f16, ov::test::ElementType::f32, ov::test::ElementType::i32};

INSTANTIATE_TEST_CASE_P(
    smoke_Subtract,
    CudaEltwiseLayerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(smoke_shapes)),
                           ::testing::Values(ngraph::helpers::EltwiseTypes::SUBTRACT),
                           ::testing::ValuesIn(input_layer_types),
                           ::testing::ValuesIn(smoke_op_types),
                           ::testing::ValuesIn(sub_precisions),
                           ::testing::Values(ov::test::ElementType::undefined),
                           ::testing::Values(ov::test::ElementType::undefined),
                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                           ::testing::Values(additional_config)),
        ::testing::Values(OperationMode::NORMAL)),
    CudaEltwiseLayerTest::getTestCaseName);

const std::vector<ov::test::ElementType> div_precisions = {
    ov::test::ElementType::f16, ov::test::ElementType::f32, ov::test::ElementType::i32};

INSTANTIATE_TEST_CASE_P(
    smoke_Divide,
    CudaEltwiseLayerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(smoke_shapes)),
                           ::testing::Values(ngraph::helpers::EltwiseTypes::DIVIDE),
                           ::testing::ValuesIn(input_layer_types),
                           ::testing::ValuesIn(smoke_op_types),
                           ::testing::ValuesIn(div_precisions),
                           ::testing::Values(ov::test::ElementType::undefined),
                           ::testing::Values(ov::test::ElementType::undefined),
                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                           ::testing::Values(additional_config)),
        ::testing::ValuesIn({OperationMode::NORMAL, OperationMode::PYTHON_DIVIDE})),
    CudaEltwiseLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    smoke_Divide_U32,
    CudaEltwiseLayerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(smoke_shapes)),
                           ::testing::Values(ngraph::helpers::EltwiseTypes::DIVIDE),
                           ::testing::ValuesIn(input_layer_types),
                           ::testing::ValuesIn(smoke_op_types),
                           ::testing::Values(ov::test::ElementType::u32),
                           ::testing::Values(ov::test::ElementType::u32),
                           ::testing::Values(ov::test::ElementType::u32),
                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                           ::testing::Values(additional_config)),
        ::testing::ValuesIn({OperationMode::NORMAL, OperationMode::PYTHON_DIVIDE})),
    CudaEltwiseLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    smoke_Divide_I64,
    CudaEltwiseLayerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(smoke_shapes)),
                           ::testing::Values(ngraph::helpers::EltwiseTypes::DIVIDE),
                           ::testing::ValuesIn(input_layer_types),
                           ::testing::ValuesIn(smoke_op_types),
                           ::testing::Values(ov::test::ElementType::i64),
                           ::testing::Values(ov::test::ElementType::i64),
                           ::testing::Values(ov::test::ElementType::i64),
                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                           ::testing::Values(additional_config)),
        ::testing::ValuesIn({OperationMode::NORMAL, OperationMode::PYTHON_DIVIDE})),
    CudaEltwiseLayerTest::getTestCaseName);

const std::vector<ov::test::ElementType> sq_diff_precisions = {ov::test::ElementType::f16,
                                                               ov::test::ElementType::f32,
                                                               ov::test::ElementType::i32,
                                                               ov::test::ElementType::i16,
                                                               ov::test::ElementType::u8};

INSTANTIATE_TEST_CASE_P(
    smoke_SquaredDifference,
    CudaEltwiseLayerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(smoke_shapes)),
                           ::testing::Values(ngraph::helpers::EltwiseTypes::SQUARED_DIFF),
                           ::testing::ValuesIn(input_layer_types),
                           ::testing::ValuesIn(smoke_op_types),
                           ::testing::ValuesIn(sq_diff_precisions),
                           ::testing::Values(ov::test::ElementType::undefined),
                           ::testing::Values(ov::test::ElementType::undefined),
                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                           ::testing::Values(additional_config)),
        ::testing::Values(OperationMode::NORMAL)),
    CudaEltwiseLayerTest::getTestCaseName);

const std::vector<ov::test::ElementType> floor_mod_precisions = {
    ov::test::ElementType::f16, ov::test::ElementType::f32, ov::test::ElementType::i32, ov::test::ElementType::u8};

INSTANTIATE_TEST_CASE_P(
    smoke_FloorMod,
    CudaEltwiseLayerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(smoke_shapes)),
                           ::testing::Values(ngraph::helpers::EltwiseTypes::FLOOR_MOD),
                           ::testing::ValuesIn(input_layer_types),
                           ::testing::ValuesIn(smoke_op_types),
                           ::testing::ValuesIn(floor_mod_precisions),
                           ::testing::Values(ov::test::ElementType::undefined),
                           ::testing::Values(ov::test::ElementType::undefined),
                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                           ::testing::Values(additional_config)),
        ::testing::Values(OperationMode::NORMAL)),
    CudaEltwiseLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    smoke_FloorMod_U32,
    CudaEltwiseLayerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(smoke_shapes)),
                           ::testing::Values(ngraph::helpers::EltwiseTypes::FLOOR_MOD),
                           ::testing::ValuesIn(input_layer_types),
                           ::testing::ValuesIn(smoke_op_types),
                           ::testing::Values(ov::test::ElementType::u32),
                           ::testing::Values(ov::test::ElementType::u32),
                           ::testing::Values(ov::test::ElementType::u32),
                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                           ::testing::Values(additional_config)),
        ::testing::Values(OperationMode::NORMAL)),
    CudaEltwiseLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    smoke_FloorMod_I64,
    CudaEltwiseLayerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(smoke_shapes)),
                           ::testing::Values(ngraph::helpers::EltwiseTypes::FLOOR_MOD),
                           ::testing::ValuesIn(input_layer_types),
                           ::testing::ValuesIn(smoke_op_types),
                           ::testing::Values(ov::test::ElementType::i64),
                           ::testing::Values(ov::test::ElementType::i64),
                           ::testing::Values(ov::test::ElementType::i64),
                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                           ::testing::Values(additional_config)),
        ::testing::Values(OperationMode::NORMAL)),
    CudaEltwiseLayerTest::getTestCaseName);

const std::vector<ov::test::ElementType> mod_precisions = {
    ov::test::ElementType::f16,
    ov::test::ElementType::f32,
    ov::test::ElementType::i32,
    ov::test::ElementType::i16,
    ov::test::ElementType::u8,
};

INSTANTIATE_TEST_CASE_P(
    smoke_Mod,
    CudaEltwiseLayerTest,
    ::testing::Combine(
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(smoke_shapes)),
                           ::testing::Values(ngraph::helpers::EltwiseTypes::MOD),
                           ::testing::ValuesIn(input_layer_types),
                           ::testing::ValuesIn(smoke_op_types),
                           ::testing::ValuesIn(mod_precisions),
                           ::testing::Values(ov::test::ElementType::undefined),
                           ::testing::Values(ov::test::ElementType::undefined),
                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                           ::testing::Values(additional_config)),
        ::testing::Values(OperationMode::NORMAL)),
    CudaEltwiseLayerTest::getTestCaseName);

//
// Resnet-50 and VGG-16
//
const std::vector<std::vector<ov::Shape>> renset50_vgg16_add_shapes = {{{1, 1000}, {1, 1000}},
                                                                       {{1, 1001}, {1, 1001}},
                                                                       {{1, 1024, 14, 14}, {1, 1024, 1, 1}},
                                                                       {{1, 1024, 14, 14}, {1, 1024, 14, 14}},
                                                                       {{1, 128, 112, 112}, {1, 128, 1, 1}},
                                                                       {{1, 128, 28, 28}, {1, 128, 1, 1}},
                                                                       {{1, 128, 56, 56}, {1, 128, 1, 1}},
                                                                       {{1, 2048, 7, 7}, {1, 2048, 1, 1}},
                                                                       {{1, 2048, 7, 7}, {1, 2048, 7, 7}},
                                                                       {{1, 256, 14, 14}, {1, 256, 1, 1}},
                                                                       {{1, 256, 28, 28}, {1, 256, 1, 1}},
                                                                       {{1, 256, 56, 56}, {1, 256, 1, 1}},
                                                                       {{1, 256, 56, 56}, {1, 256, 56, 56}},
                                                                       {{1, 3, 224, 224}, {1, 3, 1, 1}},
                                                                       {{1, 4096}, {1, 4096}},
                                                                       {{1, 512, 14, 14}, {1, 512, 1, 1}},
                                                                       {{1, 512, 28, 28}, {1, 512, 1, 1}},
                                                                       {{1, 512, 28, 28}, {1, 512, 28, 28}},
                                                                       {{1, 512, 7, 7}, {1, 512, 1, 1}},
                                                                       {{1, 64, 112, 112}, {1, 64, 1, 1}},
                                                                       {{1, 64, 224, 224}, {1, 64, 1, 1}},
                                                                       {{1, 64, 56, 56}, {1, 64, 1, 1}}};

const std::vector<ov::test::ElementType> renset50_vgg16_input_precisions = {ov::test::ElementType::f16,
                                                                            ov::test::ElementType::f32};

const std::vector<CommonTestUtils::OpType> renset50_vgg16_op_types = {CommonTestUtils::OpType::VECTOR};

INSTANTIATE_TEST_CASE_P(
    renset50_vgg16_Add,
    CudaEltwiseLayerTest,
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                              renset50_vgg16_add_shapes)),
                                          ::testing::Values(ngraph::helpers::EltwiseTypes::ADD),
                                          ::testing::ValuesIn(input_layer_types),
                                          ::testing::ValuesIn(renset50_vgg16_op_types),
                                          ::testing::ValuesIn(renset50_vgg16_input_precisions),
                                          ::testing::Values(ov::test::ElementType::undefined),
                                          ::testing::Values(ov::test::ElementType::undefined),
                                          ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                                          ::testing::Values(additional_config)),
                       ::testing::Values(OperationMode::NORMAL)),
    CudaEltwiseLayerTest::getTestCaseName);

//
// Tacotron2
//
const std::vector<std::vector<ov::Shape>> tacotron2_Multiply_shapes = {
    {{1}, {1}},
    {{2}, {1}},
    {{1}, {2}},
    {{10}, {1}},
    {{1}, {10}},
};

const std::vector<ov::test::ElementType> tacotron2_Multiply_input_precisions = {
    ov::test::ElementType::i32,
};

INSTANTIATE_TEST_CASE_P(
    tacotron2_Multiply,
    CudaEltwiseLayerTest,
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                              tacotron2_Multiply_shapes)),
                                          ::testing::Values(ngraph::helpers::EltwiseTypes::MULTIPLY),
                                          ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                                          ::testing::Values(CommonTestUtils::OpType::VECTOR),
                                          ::testing::ValuesIn(tacotron2_Multiply_input_precisions),
                                          ::testing::Values(ov::test::ElementType::undefined),
                                          ::testing::Values(ov::test::ElementType::undefined),
                                          ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                                          ::testing::Values(additional_config)),
                       ::testing::Values(OperationMode::NORMAL)),
    CudaEltwiseLayerTest::getTestCaseName);

//
// Benchmark
//
const std::vector<std::vector<ov::Shape>> bench_shapes = {{{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}},
                                                          {{2, 3, 4, 5}, {2, 1, 1, 5}},
                                                          {{10, 10}, {1, 10}},
                                                          {{1, 96, 75, 75}, {1, 96, 75, 75}},
                                                          {{1, 576, 19, 19}, {1, 576, 19, 19}},
                                                          {{1, 1280, 10, 10}, {1, 1280, 10, 10}},
                                                          {{512, 192, 192, 2}, {512, 192, 192, 2}},
                                                          {{1024, 384, 384, 2}, {1024, 384, 384, 2}},
                                                          {{1024, 1024, 384, 2}, {1, 1024, 1, 2}},
                                                          {{1024, 1024, 384, 2}, {1}}};

//
// Benchmark Add
//
struct AddBenchmarkTest : BenchmarkLayerTest<CudaEltwiseLayerTest> {};

TEST_P(AddBenchmarkTest, DISABLED_Add_Benchmark) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run("Add", std::chrono::milliseconds(2000), 100);
}

const std::vector<ov::test::ElementType> bench_add_precisions = {ov::test::ElementType::f16,
                                                                 ov::test::ElementType::f32};

INSTANTIATE_TEST_CASE_P(
    Add_Benchmark,
    AddBenchmarkTest,
    ::testing::Combine(
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(bench_shapes)),
                           ::testing::Values(ngraph::helpers::EltwiseTypes::ADD),
                           ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                           ::testing::Values(CommonTestUtils::OpType::VECTOR),
                           ::testing::ValuesIn(bench_add_precisions),
                           ::testing::Values(ov::test::ElementType::undefined),
                           ::testing::Values(ov::test::ElementType::undefined),
                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                           ::testing::Values(additional_config)),
        ::testing::Values(OperationMode::NORMAL)),
    CudaEltwiseLayerTest::getTestCaseName);

//
// Benchmark Multiply
//
struct MultiplyBenchmarkTest : BenchmarkLayerTest<CudaEltwiseLayerTest> {};

TEST_P(MultiplyBenchmarkTest, DISABLED_Multiply_Benchmark) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run("Multiply", std::chrono::milliseconds(2000), 100);
}

const std::vector<ov::test::ElementType> bench_mul_precisions = {ov::test::ElementType::f16,
                                                                 ov::test::ElementType::f32};

INSTANTIATE_TEST_CASE_P(
    Multiply_Benchmark,
    MultiplyBenchmarkTest,
    ::testing::Combine(
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(bench_shapes)),
                           ::testing::Values(ngraph::helpers::EltwiseTypes::MULTIPLY),
                           ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                           ::testing::Values(CommonTestUtils::OpType::VECTOR),
                           ::testing::ValuesIn(bench_mul_precisions),
                           ::testing::Values(ov::test::ElementType::undefined),
                           ::testing::Values(ov::test::ElementType::undefined),
                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                           ::testing::Values(additional_config)),
        ::testing::Values(OperationMode::NORMAL)),
    CudaEltwiseLayerTest::getTestCaseName);

//
// Benchmark Subtract
//
struct SubtractBenchmarkTest : BenchmarkLayerTest<CudaEltwiseLayerTest> {};

TEST_P(SubtractBenchmarkTest, DISABLED_Subtract_Benchmark) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run("Subtract", std::chrono::milliseconds(2000), 10);
}

const std::vector<ov::test::ElementType> bench_sub_precisions = {
    ov::test::ElementType::f16, ov::test::ElementType::f32, ov::test::ElementType::i32};

INSTANTIATE_TEST_CASE_P(
    Subtract_Benchmark,
    SubtractBenchmarkTest,
    ::testing::Combine(
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(bench_shapes)),
                           ::testing::Values(ngraph::helpers::EltwiseTypes::SUBTRACT),
                           ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                           ::testing::Values(CommonTestUtils::OpType::VECTOR),
                           ::testing::ValuesIn(bench_sub_precisions),
                           ::testing::Values(ov::test::ElementType::undefined),
                           ::testing::Values(ov::test::ElementType::undefined),
                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                           ::testing::Values(additional_config)),
        ::testing::Values(OperationMode::NORMAL)),
    CudaEltwiseLayerTest::getTestCaseName);

//
// Benchmark Divide
//
struct DivideBenchmarkTest : BenchmarkLayerTest<CudaEltwiseLayerTest> {};

TEST_P(DivideBenchmarkTest, DISABLED_Divide_Benchmark) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run("Divide", std::chrono::milliseconds(2000), 10);
}

const std::vector<ov::test::ElementType> bench_div_precisions = {
    ov::test::ElementType::f16, ov::test::ElementType::f32, ov::test::ElementType::i32};

INSTANTIATE_TEST_CASE_P(
    Divide_Benchmark,
    DivideBenchmarkTest,
    ::testing::Combine(
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(bench_shapes)),
                           ::testing::Values(ngraph::helpers::EltwiseTypes::DIVIDE),
                           ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                           ::testing::Values(CommonTestUtils::OpType::VECTOR),
                           ::testing::ValuesIn(bench_div_precisions),
                           ::testing::Values(ov::test::ElementType::undefined),
                           ::testing::Values(ov::test::ElementType::undefined),
                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                           ::testing::Values(additional_config)),
        ::testing::ValuesIn({OperationMode::NORMAL, OperationMode::PYTHON_DIVIDE})),
    CudaEltwiseLayerTest::getTestCaseName);

//
// Benchmark Mod
//
struct ModBenchmarkTest : BenchmarkLayerTest<CudaEltwiseLayerTest> {};

TEST_P(ModBenchmarkTest, DISABLED_Mod_Benchmark) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run("Mod", std::chrono::milliseconds(2000), 10);
}

const std::vector<ov::test::ElementType> bench_mod_precisions = {
    ov::test::ElementType::f16,
    ov::test::ElementType::f32,
    ov::test::ElementType::i32,
    ov::test::ElementType::i16,
    ov::test::ElementType::u8,
};

INSTANTIATE_TEST_CASE_P(
    Mod_Benchmark,
    ModBenchmarkTest,
    ::testing::Combine(
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(bench_shapes)),
                           ::testing::Values(ngraph::helpers::EltwiseTypes::MOD),
                           ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                           ::testing::Values(CommonTestUtils::OpType::VECTOR),
                           ::testing::ValuesIn(bench_mod_precisions),
                           ::testing::Values(ov::test::ElementType::undefined),
                           ::testing::Values(ov::test::ElementType::undefined),
                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                           ::testing::Values(additional_config)),
        ::testing::Values(OperationMode::NORMAL)),
    CudaEltwiseLayerTest::getTestCaseName);

}  // namespace
}  // namespace LayerTestsDefinitions