// clang-format off
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_tests_instances/test_utils.hpp"
#include "integration/test_constants.hpp"

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <utility>
#include <vector>

#include "behavior/ov_infer_request/batched_tensors.hpp"
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

namespace ov::test::behavior {

using OVInferRequestCallbackTests = OVInferRequestTests;

TEST_P(OVInferRequestCallbackTests, canCallAsyncWithCompletionCallback) {
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    bool is_called = false;
    OV_ASSERT_NO_THROW(req.set_callback([&](std::exception_ptr exception_ptr) {
        ASSERT_EQ(exception_ptr, nullptr);
        is_called = true;
    }));
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    ASSERT_TRUE(is_called);
}

TEST_P(OVInferRequestCallbackTests, syncInferDoesNotCallCompletionCallback) {
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    bool is_called = false;
    req.set_callback([&](std::exception_ptr exception_ptr) {
        ASSERT_EQ(nullptr, exception_ptr);
        is_called = true;
    });
    req.infer();
    ASSERT_FALSE(is_called);
}

TEST_P(OVInferRequestCallbackTests, canStartSeveralAsyncInsideCompletionCallbackWithSafeDtor) {
    constexpr int num_iter = 10;
    struct TestUserData {
        std::atomic<int> num_iter_seen = {0};
        std::promise<bool> promise;
    };
    TestUserData data;

    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req.set_callback([&](std::exception_ptr exception_ptr) {
        if (exception_ptr) {
            data.promise.set_exception(exception_ptr);
            return;
        }
        if (data.num_iter_seen.fetch_add(1) != num_iter) {
            req.start_async();
        } else {
            data.promise.set_value(true);
        }
    }));
    auto future = data.promise.get_future();
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    future.wait();
    ASSERT_TRUE(future.get());
    ASSERT_EQ(num_iter, data.num_iter_seen - 1);
}

TEST_P(OVInferRequestCallbackTests, returnGeneralErrorIfCallbackThrowException) {
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req.set_callback([](std::exception_ptr) {
        OPENVINO_THROW("Throw");
    }));
    OV_ASSERT_NO_THROW(req.start_async());
    ASSERT_THROW(req.wait(), ov::Exception);
}

TEST_P(OVInferRequestCallbackTests, ReturnResultNotReadyFromWaitInAsyncModeForTooSmallTimeout) {
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    std::promise<std::chrono::system_clock::time_point> callback_time_stamp;
    auto callback_time_stamp_future = callback_time_stamp.get_future();
    OV_ASSERT_NO_THROW(req.set_callback([&](std::exception_ptr exception_ptr) {
        if (exception_ptr) {
            callback_time_stamp.set_exception(exception_ptr);
        } else {
            callback_time_stamp.set_value(std::chrono::system_clock::now());
        }
    }));
    OV_ASSERT_NO_THROW(req.start_async());
    bool ready = false;
    OV_ASSERT_NO_THROW(ready = req.wait_for({}));
    const auto after_wait_time_stamp = std::chrono::system_clock::now();
    if (after_wait_time_stamp < callback_time_stamp_future.get()) {
        ASSERT_FALSE(ready);
    }
    OV_ASSERT_NO_THROW(req.wait());
}

TEST_P(OVInferRequestCallbackTests, ImplDoesNotCopyCallback) {
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    {
        auto some_ptr = std::make_shared<int>(42);
        OV_ASSERT_NO_THROW(req.set_callback([some_ptr](std::exception_ptr exception_ptr) {
            ASSERT_EQ(nullptr, exception_ptr);
            ASSERT_EQ(1, some_ptr.use_count());
        }));
    }
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
}

}  // namespace ov::test::behavior

using ov::test::behavior::OVInferRequestCallbackTests;

namespace {

const std::vector<ov::AnyMap> empty_configs = {{}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         InferRequestPropertiesTest,
                         ::testing::Combine(::testing::Values(1u),
                                            ::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(empty_configs)),
                         InferRequestPropertiesTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVInferRequestCallbackTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(empty_configs)),
                         OVInferRequestCallbackTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVInferRequestWaitTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(empty_configs)),
                         OVInferRequestWaitTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVInferRequestCancellationTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(empty_configs)),
                         OVInferRequestCancellationTests::getTestCaseName);

std::vector<ov::element::Type> prcs = {
    ov::element::boolean, ov::element::bf16, ov::element::f16, ov::element::f32, ov::element::f64, ov::element::i4,
    ov::element::i8,      ov::element::i16, ov::element::i32, ov::element::i64, ov::element::u1,  ov::element::u4,
    ov::element::u8,      ov::element::u16, ov::element::u32, ov::element::u64,
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVInferRequestIOTensorTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(empty_configs)),
                         OVInferRequestIOTensorTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVInferRequestCheckTensorPrecision,
                         ::testing::Combine(::testing::ValuesIn(prcs),
                                            ::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(empty_configs)),
                         OVInferRequestCheckTensorPrecision::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVInferRequestBatchedTests,
                         ::testing::Values(ov::test::utils::DEVICE_GFX),
                         OVInferRequestBatchedTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVInferRequestInferenceTests,
                         ::testing::Combine(::testing::Values(ov::test::behavior::tensor_roi::roi_nchw(),
                                                              ov::test::behavior::tensor_roi::roi_1d()),
                                            ::testing::Values(ov::test::utils::DEVICE_GFX)),
                         OVInferRequestInferenceTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVInferenceChaining,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(empty_configs)),
                         OVInferenceChaining::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVInferenceChainingStatic,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(empty_configs)),
                         OVInferenceChainingStatic::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_BehaviorTests,
    OVInferRequestDynamicTests,
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
                         OVInferRequestVariableStateTest,
                         ::testing::ValuesIn(memoryStateTestCases),
                         OVInferRequestVariableStateTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVInferRequestMultithreadingTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(empty_configs)),
                         OVInferRequestMultithreadingTests::getTestCaseName);

}  // namespace
