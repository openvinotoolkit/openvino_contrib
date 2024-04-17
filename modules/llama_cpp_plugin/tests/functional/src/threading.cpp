// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <thread>

#include "benchmarking.hpp"
#include "llm_inference.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"

const std::string MODEL_FILE = ov::test::utils::getCurrentWorkingDir() + SEP + TEST_FILES_DIR + SEP + "gpt2.gguf";
constexpr size_t NUM_INFER_REQUESTS_FOR_BENCHMARK = 256;

enum class ThreadSettingType { PLUGIN = 0, MODEL = 1, PLUGIN_AND_MODEL = 2 };

class LlamaCppThreadSettingTypeTest : public testing::TestWithParam<ThreadSettingType> {};

ov::CompiledModel get_model_with_thread_setting(int num_threads, ThreadSettingType thread_setting_type) {
    ov::Core core;
    if (thread_setting_type == ThreadSettingType::PLUGIN ||
        thread_setting_type == ThreadSettingType::PLUGIN_AND_MODEL) {
        core.set_property("LLAMA_CPP", ov::AnyMap{{ov::inference_num_threads.name(), num_threads}});
    }

    if (thread_setting_type == ThreadSettingType::MODEL || thread_setting_type == ThreadSettingType::PLUGIN_AND_MODEL) {
        return core.compile_model(MODEL_FILE, "LLAMA_CPP", ov::AnyMap{{ov::inference_num_threads.name(), num_threads}});
    }

    return core.compile_model(MODEL_FILE, "LLAMA_CPP");
}

void infer_one_token_fn(ov::InferRequest& infer_request) {
    infer_and_get_last_logits(infer_request, {1337}, 0);
}

double measure_inference_speed_for_thread_count(int num_threads, ThreadSettingType thread_setting_type) {
    auto model = get_model_with_thread_setting(num_threads, thread_setting_type);

    auto infer_request = model.create_infer_request();

    auto infer_one_token_lambda = [&infer_request](void) -> void {
        infer_one_token_fn(infer_request);
    };
    return measure_iterations_per_second(infer_one_token_lambda, NUM_INFER_REQUESTS_FOR_BENCHMARK);
}

TEST_P(LlamaCppThreadSettingTypeTest, NumThreadSettingDoesntFail) {
    ThreadSettingType thread_setting_type = GetParam();
    constexpr size_t NUM_THREADS_TO_SET = 2;

    auto model = get_model_with_thread_setting(NUM_THREADS_TO_SET, thread_setting_type);

    auto infer_request = model.create_infer_request();
    std::vector<int64_t> mock_input_ids{1337, NUM_THREADS_TO_SET * 10};
    infer_and_get_last_logits(infer_request, mock_input_ids, 0);
}

TEST_P(LlamaCppThreadSettingTypeTest, ThreadedExecutionIsFaster) {
    ThreadSettingType thread_setting_type = GetParam();
    double single_threaded_iters_per_second = measure_inference_speed_for_thread_count(1, thread_setting_type);

    size_t optimal_num_threads = std::thread::hardware_concurrency();
    ASSERT_GT(optimal_num_threads, 1);

    double multi_threaded_iters_per_second =
        measure_inference_speed_for_thread_count(optimal_num_threads, thread_setting_type);
    std::cout << "threaded " << multi_threaded_iters_per_second << ", non-threaded " << single_threaded_iters_per_second
              << std::endl;
    ASSERT_GE(multi_threaded_iters_per_second / single_threaded_iters_per_second, 1.1);
}

INSTANTIATE_TEST_SUITE_P(CheckForAllThreadSettingTypes,
                         LlamaCppThreadSettingTypeTest,
                         ::testing::Values(ThreadSettingType::PLUGIN,
                                           ThreadSettingType::MODEL,
                                           ThreadSettingType::PLUGIN_AND_MODEL));

TEST(LlamaCppConsecutiveThreadSettingTest, ConsecutiveThreadSettingUsesModelValue) {
    double ref_two_thread_iters_per_second = measure_inference_speed_for_thread_count(2, ThreadSettingType::MODEL);

    ov::Core core;
    core.set_property("LLAMA_CPP", ov::inference_num_threads(1));
    auto model = core.compile_model(MODEL_FILE, "LLAMA_CPP", ov::AnyMap{{ov::inference_num_threads.name(), 2}});
    auto infer_request = model.create_infer_request();

    double test_two_thread_iters_per_second_via_consecutive_setting = measure_iterations_per_second(
        [&infer_request](void) -> void {
            infer_one_token_fn(infer_request);
        },
        NUM_INFER_REQUESTS_FOR_BENCHMARK);

    size_t optimal_num_threads = std::thread::hardware_concurrency();
    ASSERT_GT(optimal_num_threads, 1);

    EXPECT_NEAR(test_two_thread_iters_per_second_via_consecutive_setting,
                ref_two_thread_iters_per_second,
                0.05 * ref_two_thread_iters_per_second);
}
