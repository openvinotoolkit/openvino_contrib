// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/comparison.hpp"

#include <vector>

#include "cuda_test_constants.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "ov_unsymmetrical_comparer.hpp"
namespace {
using namespace ov::test;
using namespace ov::test::utils;

class UnsymmetricalComparisonLayerTest : public UnsymmetricalComparer<ComparisonLayerTest> {};

TEST_P(UnsymmetricalComparisonLayerTest, Inference) { run(); }

std::map<ov::Shape, std::vector<ov::Shape>> smoke_shapes = {
    {{1}, {{1}, {17}, {1, 1}, {2, 18}, {1, 1, 2}, {2, 2, 3}, {1, 1, 2, 3}}},
    {{5}, {{1}, {1, 1}, {2, 5}, {1, 1, 1}, {2, 2, 5}}},
    {{2, 200}, {{1}, {200}, {1, 200}, {2, 200}, {2, 2, 200}}},
    {{1, 3, 20}, {{20}, {2, 1, 1}}},
    {{2, 17, 3, 4}, {{4}, {1, 3, 4}, {2, 1, 3, 4}}},
    {{2, 1, 1, 3, 1}, {{1}, {1, 3, 4}, {2, 1, 3, 4}, {1, 1, 1, 1, 1}}},
};
auto converter = [] (const std::vector<std::pair<ov::Shape, ov::Shape>>& shapes) {
    std::vector<std::vector<ov::Shape>> result;
    for (const auto& shape : shapes) {
        result.push_back({shape.first, shape.second});
    }
    return result;
};
auto input_smoke_shapes_pair_vector = ov::test::utils::combineParams(smoke_shapes);
auto input_smoke_shapes_static = converter(input_smoke_shapes_pair_vector);

std::map<ov::Shape, std::vector<ov::Shape>> shapes = {
    {{1}, {{256}}},
};
auto input_shapes_pair_vector = ov::test::utils::combineParams(shapes);
auto input_shapes_static = converter(input_shapes_pair_vector);

std::vector<ov::element::Type> model_type = {
    ov::element::f32,
    ov::element::f16,
};

std::vector<ComparisonTypes> comparison_op_types = {
    ComparisonTypes::EQUAL,
    ComparisonTypes::NOT_EQUAL,
    ComparisonTypes::GREATER,
    ComparisonTypes::GREATER_EQUAL,
    ComparisonTypes::LESS,
    ComparisonTypes::LESS_EQUAL,
};

std::vector<InputLayerType> second_input_types = {
    InputLayerType::CONSTANT,
    InputLayerType::PARAMETER,
};

std::map<std::string, std::string> additional_config = {};

const auto smoke_comparison_test_params =
    ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_smoke_shapes_static)),
                       ::testing::ValuesIn(comparison_op_types),
                       ::testing::ValuesIn(second_input_types),
                       ::testing::ValuesIn(model_type),
                       ::testing::Values(DEVICE_NVIDIA),
                       ::testing::Values(additional_config));

const auto comparison_test_params = ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                                                       ::testing::ValuesIn(comparison_op_types),
                                                       ::testing::ValuesIn(second_input_types),
                                                       ::testing::ValuesIn(model_type),
                                                       ::testing::Values(DEVICE_NVIDIA),
                                                       ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(smoke_ComparisonCompareWithRefs,
                        UnsymmetricalComparisonLayerTest,
                        smoke_comparison_test_params,
                        ComparisonLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(ComparisonCompareWithRefs,
                        UnsymmetricalComparisonLayerTest,
                        comparison_test_params,
                        ComparisonLayerTest::getTestCaseName);

}  // namespace

// ------------- Benchmark -------------
#include "ov_benchmark.hpp"

namespace {
namespace benchmark {
struct ComparisonBenchmarkTest : BenchmarkLayerTest<ComparisonLayerTest> {};

TEST_P(ComparisonBenchmarkTest, DISABLED_benchmark) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run("Comparison", std::chrono::milliseconds(2000), 100);
}

INSTANTIATE_TEST_CASE_P(smoke_ComparisonCompareWithRefs,
                        ComparisonBenchmarkTest,
                        smoke_comparison_test_params,
                        ComparisonLayerTest::getTestCaseName);

}  // namespace benchmark
}  // namespace