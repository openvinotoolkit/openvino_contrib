// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "model_fixture.hpp"

// "Why is the Sun yellow?"
const std::vector<int64_t> GPT2_PROMPT_TOKEN_IDS = {5195, 318, 262, 3825, 7872, 30};
// "The Sun is a bright red, which means it is a bright red. The Sun is a bright
// red because it is a bright red."
const std::vector<int64_t> GPT2_REFERENCE_RESPONSE_TOKEN_IDS = {
    198, 464,  3825, 318, 257,  6016, 2266, 11,  543, 1724, 340,  318,  257, 6016, 2266, 13,
    383, 3825, 318,  257, 6016, 2266, 780,  340, 318, 257,  6016, 2266, 13,  198,  198,  464};

TEST_F(CompiledModelTest, TestPromptResponseGPT2) {
    ov::InferRequest lm = model.create_infer_request();
    auto input_ids_tensor = ov::Tensor(ov::element::Type_t::i64, {1, GPT2_PROMPT_TOKEN_IDS.size()});
    std::copy(GPT2_PROMPT_TOKEN_IDS.begin(), GPT2_PROMPT_TOKEN_IDS.end(), input_ids_tensor.data<int64_t>());
    lm.set_tensor("input_ids", input_ids_tensor);
    ov::Tensor position_ids = lm.get_tensor("position_ids");
    position_ids.set_shape(input_ids_tensor.get_shape());
    std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + position_ids.get_size(), 0);

    fill_unused_inputs(lm, input_ids_tensor.get_shape());

    lm.infer();

    size_t vocab_size = lm.get_tensor("logits").get_shape().back();
    float* logits = lm.get_tensor("logits").data<float>() + (input_ids_tensor.get_size() - 1) * vocab_size;
    int64_t out_token = std::max_element(logits, logits + vocab_size) - logits;

    constexpr size_t BATCH_SIZE = 1;
    lm.get_tensor("input_ids").set_shape({BATCH_SIZE, 1});
    position_ids.set_shape({BATCH_SIZE, 1});

    size_t cnt = 0;
    std::vector<int64_t> out_token_ids;

    while (cnt < GPT2_REFERENCE_RESPONSE_TOKEN_IDS.size()) {
        lm.get_tensor("input_ids").data<int64_t>()[0] = out_token;
        lm.get_tensor("attention_mask").set_shape({BATCH_SIZE, lm.get_tensor("attention_mask").get_shape().at(1) + 1});
        std::fill_n(lm.get_tensor("attention_mask").data<int64_t>(), lm.get_tensor("attention_mask").get_size(), 1);
        position_ids.data<int64_t>()[0] = int64_t(lm.get_tensor("attention_mask").get_size() - 2);
        lm.start_async();
        lm.wait();
        logits = lm.get_tensor("logits").data<float>();
        out_token = std::max_element(logits, logits + vocab_size) - logits;
        out_token_ids.push_back(out_token);
        cnt++;
    }

    lm.reset_state();

    ASSERT_EQ(out_token_ids, GPT2_REFERENCE_RESPONSE_TOKEN_IDS);
}
