// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <single_layer_tests/activation.hpp>
#include <cuda_test_constants.hpp>

namespace LayerTestsDefinitions {
namespace {

std::initializer_list<std::initializer_list<std::size_t>> reluShapes{
    {1, 1024, 14, 14}, {1, 128, 112, 112}, {1, 128, 28, 28}, {1, 128, 56, 56},
    {1, 2048, 7, 7},   {1, 256, 14, 14},   {1, 256, 28, 28}, {1, 256, 56, 56},
    {1, 4096},         {1, 512, 14, 14},   {1, 512, 28, 28}, {1, 512, 7, 7},
    {1, 64, 112, 112}, {1, 64, 224, 224},  {1, 64, 56, 56},
};
auto listToVectors(
    const std::initializer_list<std::initializer_list<std::size_t>>& list) {
  // pair of {input shape, unused}
  std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> shapes;
  shapes.reserve(list.size());
  for (auto& e : list) shapes.emplace_back(e, 0);
  return shapes;
}
const auto basicReluCases = ::testing::Combine(
    ::testing::Values(std::pair<ngraph::helpers::ActivationTypes, float>{
        ngraph::helpers::Relu, 0}),  // float is unused in relu
    ::testing::Values(InferenceEngine::Precision::FP32,
                      InferenceEngine::Precision::FP16),
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    ::testing::Values(InferenceEngine::Layout::ANY),
    ::testing::Values(InferenceEngine::Layout::ANY),
    ::testing::ValuesIn(listToVectors(reluShapes)),
    ::testing::Values(CommonTestUtils::DEVICE_CUDA));

INSTANTIATE_TEST_CASE_P(Activation_Basic_Relu,
		ActivationLayerTest,
		basicReluCases,
		ActivationLayerTest::getTestCaseName);

}  // namespace
}  // namespace LayerTestsDefinitions
