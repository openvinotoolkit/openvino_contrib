// clang-format off
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.hpp"
#include "integration/test_constants.hpp"

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

using GfxInferRequestPropertiesTest = ov::test::utils::GfxSkippedTests<InferRequestPropertiesTest>;
using GfxOVInferRequestBatchedTests = ov::test::utils::GfxSkippedTests<OVInferRequestBatchedTests>;
using GfxOVInferRequestCallbackTests = ov::test::utils::GfxSkippedTests<OVInferRequestCallbackTests>;
using GfxOVInferRequestCancellationTests = ov::test::utils::GfxSkippedTests<OVInferRequestCancellationTests>;
using GfxOVInferRequestIOTensorTest = ov::test::utils::GfxSkippedTests<OVInferRequestIOTensorTest>;
using GfxOVInferRequestInferenceTests = ov::test::utils::GfxSkippedTests<OVInferRequestInferenceTests>;
using GfxOVInferenceChaining = ov::test::utils::GfxSkippedTests<OVInferenceChaining>;
using GfxOVInferenceChainingStatic = ov::test::utils::GfxSkippedTests<OVInferenceChainingStatic>;
using GfxOVInferRequestDynamicTests = ov::test::utils::GfxSkippedTests<OVInferRequestDynamicTests>;
using GfxOVInferRequestWaitTests = ov::test::utils::GfxSkippedTests<OVInferRequestWaitTests>;
using GfxOVInferRequestVariableStateTest = ov::test::utils::GfxSkippedTests<OVInferRequestVariableStateTest>;
using GfxOVInferRequestMultithreadingTests = ov::test::utils::GfxSkippedTests<OVInferRequestMultithreadingTests>;
using GfxOVInferRequestCheckTensorPrecision =
    ov::test::utils::GfxSkippedTests<OVInferRequestCheckTensorPrecision>;

namespace {

const std::vector<ov::AnyMap> empty_configs = {{}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         GfxInferRequestPropertiesTest,
                         ::testing::Combine(::testing::Values(1u),
                                            ::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(empty_configs)),
                         InferRequestPropertiesTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         GfxOVInferRequestCallbackTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(empty_configs)),
                         OVInferRequestCallbackTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         GfxOVInferRequestWaitTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(empty_configs)),
                         OVInferRequestWaitTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         GfxOVInferRequestCancellationTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(empty_configs)),
                         OVInferRequestCancellationTests::getTestCaseName);

std::vector<ov::element::Type> prcs = {
    ov::element::boolean, ov::element::bf16, ov::element::f16, ov::element::f32, ov::element::f64, ov::element::i4,
    ov::element::i8,      ov::element::i16, ov::element::i32, ov::element::i64, ov::element::u1,  ov::element::u4,
    ov::element::u8,      ov::element::u16, ov::element::u32, ov::element::u64,
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         GfxOVInferRequestIOTensorTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(empty_configs)),
                         OVInferRequestIOTensorTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         GfxOVInferRequestCheckTensorPrecision,
                         ::testing::Combine(::testing::ValuesIn(prcs),
                                            ::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(empty_configs)),
                         OVInferRequestCheckTensorPrecision::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         GfxOVInferRequestBatchedTests,
                         ::testing::Values(ov::test::utils::DEVICE_GFX),
                         OVInferRequestBatchedTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         GfxOVInferRequestInferenceTests,
                         ::testing::Combine(::testing::Values(ov::test::behavior::tensor_roi::roi_nchw(),
                                                              ov::test::behavior::tensor_roi::roi_1d()),
                                            ::testing::Values(ov::test::utils::DEVICE_GFX)),
                         OVInferRequestInferenceTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         GfxOVInferenceChaining,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(empty_configs)),
                         OVInferenceChaining::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         GfxOVInferenceChainingStatic,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(empty_configs)),
                         OVInferenceChainingStatic::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_BehaviorTests,
    GfxOVInferRequestDynamicTests,
    ::testing::Combine(::testing::Values(ov::test::utils::make_split_conv_concat()),
                       ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{
                           {{1, 4, 20, 20}, {1, 10, 18, 18}},
                           {{2, 4, 20, 20}, {2, 10, 18, 18}}}),
                       ::testing::Values(ov::test::utils::DEVICE_GFX),
                       ::testing::ValuesIn(empty_configs)),
    OVInferRequestDynamicTests::getTestCaseName);

std::vector<ov::test::behavior::memoryStateParams> memoryStateTestCases = {
    ov::test::behavior::memoryStateParams(OVInferRequestVariableStateTest::get_network(),
                                          {"c_1-3", "r_1-3"},
                                          ov::test::utils::DEVICE_GFX,
                                          {})};

INSTANTIATE_TEST_SUITE_P(smoke_Template_BehaviorTests,
                         GfxOVInferRequestVariableStateTest,
                         ::testing::ValuesIn(memoryStateTestCases),
                         OVInferRequestVariableStateTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         GfxOVInferRequestMultithreadingTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(empty_configs)),
                         OVInferRequestMultithreadingTests::getTestCaseName);

}  // namespace
