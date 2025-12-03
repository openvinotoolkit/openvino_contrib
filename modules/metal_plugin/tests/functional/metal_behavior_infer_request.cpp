// clang-format off
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "metal_test_utils.hpp"
#include "test_constants.hpp"

#include <utility>
#include <vector>

#include "behavior/ov_infer_request/batched_tensors.hpp"
#include "behavior/ov_infer_request/callback.hpp"
#include "behavior/ov_infer_request/cancellation.hpp"
#include "behavior/ov_infer_request/inference.hpp"
#include "behavior/ov_infer_request/inference_chaining.hpp"
#include "behavior/ov_infer_request/infer_request_dynamic.hpp"
#include "behavior/ov_infer_request/io_tensor.hpp"
#include "behavior/ov_infer_request/memory_states.hpp"
#include "behavior/ov_infer_request/multithreading.hpp"
#include "behavior/ov_infer_request/properties_tests.hpp"
#include "behavior/ov_infer_request/wait.hpp"
#include "common_test_utils/subgraph_builders/split_conv_concat.hpp"

using ov::test::behavior::InferRequestPropertiesTest;
using ov::test::behavior::OVInferRequestBatchedTests;
using ov::test::behavior::OVInferRequestCallbackTests;
using ov::test::behavior::OVInferRequestCancellationTests;
using ov::test::behavior::OVInferRequestIOTensorTest;
using ov::test::behavior::OVInferRequestInferenceTests;
using ov::test::behavior::OVInferenceChaining;
using ov::test::behavior::OVInferenceChainingStatic;
using ov::test::behavior::OVInferRequestDynamicTests;
using ov::test::behavior::OVInferRequestWaitTests;
using ov::test::behavior::OVInferRequestVariableStateTest;
using ov::test::behavior::OVInferRequestMultithreadingTests;
using ov::test::behavior::OVInferRequestCheckTensorPrecision;

using MetalInferRequestPropertiesTest = ov::test::utils::MetalSkippedTests<InferRequestPropertiesTest>;
using MetalOVInferRequestBatchedTests = ov::test::utils::MetalSkippedTests<OVInferRequestBatchedTests>;
using MetalOVInferRequestCallbackTests = ov::test::utils::MetalSkippedTests<OVInferRequestCallbackTests>;
using MetalOVInferRequestCancellationTests = ov::test::utils::MetalSkippedTests<OVInferRequestCancellationTests>;
using MetalOVInferRequestIOTensorTest = ov::test::utils::MetalSkippedTests<OVInferRequestIOTensorTest>;
using MetalOVInferRequestInferenceTests = ov::test::utils::MetalSkippedTests<OVInferRequestInferenceTests>;
using MetalOVInferenceChaining = ov::test::utils::MetalSkippedTests<OVInferenceChaining>;
using MetalOVInferenceChainingStatic = ov::test::utils::MetalSkippedTests<OVInferenceChainingStatic>;
using MetalOVInferRequestDynamicTests = ov::test::utils::MetalSkippedTests<OVInferRequestDynamicTests>;
using MetalOVInferRequestWaitTests = ov::test::utils::MetalSkippedTests<OVInferRequestWaitTests>;
using MetalOVInferRequestVariableStateTest = ov::test::utils::MetalSkippedTests<OVInferRequestVariableStateTest>;
using MetalOVInferRequestMultithreadingTests = ov::test::utils::MetalSkippedTests<OVInferRequestMultithreadingTests>;
using MetalOVInferRequestCheckTensorPrecision =
    ov::test::utils::MetalSkippedTests<OVInferRequestCheckTensorPrecision>;

namespace {

const std::vector<ov::AnyMap> empty_configs = {{}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         MetalInferRequestPropertiesTest,
                         ::testing::Combine(::testing::Values(1u),
                                            ::testing::Values(ov::test::utils::DEVICE_METAL),
                                            ::testing::ValuesIn(empty_configs)),
                         InferRequestPropertiesTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         MetalOVInferRequestCallbackTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_METAL),
                                            ::testing::ValuesIn(empty_configs)),
                         OVInferRequestCallbackTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         MetalOVInferRequestWaitTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_METAL),
                                            ::testing::ValuesIn(empty_configs)),
                         OVInferRequestWaitTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         MetalOVInferRequestCancellationTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_METAL),
                                            ::testing::ValuesIn(empty_configs)),
                         OVInferRequestCancellationTests::getTestCaseName);

std::vector<ov::element::Type> prcs = {
    ov::element::boolean, ov::element::bf16, ov::element::f16, ov::element::f32, ov::element::f64, ov::element::i4,
    ov::element::i8,      ov::element::i16, ov::element::i32, ov::element::i64, ov::element::u1,  ov::element::u4,
    ov::element::u8,      ov::element::u16, ov::element::u32, ov::element::u64,
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         MetalOVInferRequestIOTensorTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_METAL),
                                            ::testing::ValuesIn(empty_configs)),
                         OVInferRequestIOTensorTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         MetalOVInferRequestCheckTensorPrecision,
                         ::testing::Combine(::testing::ValuesIn(prcs),
                                            ::testing::Values(ov::test::utils::DEVICE_METAL),
                                            ::testing::ValuesIn(empty_configs)),
                         OVInferRequestCheckTensorPrecision::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         MetalOVInferRequestBatchedTests,
                         ::testing::Values(ov::test::utils::DEVICE_METAL),
                         OVInferRequestBatchedTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         MetalOVInferRequestInferenceTests,
                         ::testing::Combine(::testing::Values(ov::test::behavior::tensor_roi::roi_nchw(),
                                                              ov::test::behavior::tensor_roi::roi_1d()),
                                            ::testing::Values(ov::test::utils::DEVICE_METAL)),
                         OVInferRequestInferenceTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         MetalOVInferenceChaining,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_METAL),
                                            ::testing::ValuesIn(empty_configs)),
                         OVInferenceChaining::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         MetalOVInferenceChainingStatic,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_METAL),
                                            ::testing::ValuesIn(empty_configs)),
                         OVInferenceChainingStatic::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_BehaviorTests,
    MetalOVInferRequestDynamicTests,
    ::testing::Combine(::testing::Values(ov::test::utils::make_split_conv_concat()),
                       ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{
                           {{1, 4, 20, 20}, {1, 10, 18, 18}},
                           {{2, 4, 20, 20}, {2, 10, 18, 18}}}),
                       ::testing::Values(ov::test::utils::DEVICE_METAL),
                       ::testing::ValuesIn(empty_configs)),
    OVInferRequestDynamicTests::getTestCaseName);

std::vector<ov::test::behavior::memoryStateParams> memoryStateTestCases = {
    ov::test::behavior::memoryStateParams(OVInferRequestVariableStateTest::get_network(),
                                          {"c_1-3", "r_1-3"},
                                          ov::test::utils::DEVICE_METAL,
                                          {})};

INSTANTIATE_TEST_SUITE_P(smoke_Template_BehaviorTests,
                         MetalOVInferRequestVariableStateTest,
                         ::testing::ValuesIn(memoryStateTestCases),
                         OVInferRequestVariableStateTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         MetalOVInferRequestMultithreadingTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_METAL),
                                            ::testing::ValuesIn(empty_configs)),
                         OVInferRequestMultithreadingTests::getTestCaseName);

}  // namespace
