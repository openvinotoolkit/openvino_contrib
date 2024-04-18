// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "llm_inference.hpp"
#include "model_fixture.hpp"
#include "openvino/runtime/infer_request.hpp"

const std::vector<int64_t> GPT2_SUN_PROMPT_TOKEN_IDS = {5195, 318, 262, 3825, 7872, 30};
const std::vector<int64_t> GPT2_LENNON_PROMPT_TOKEN_IDS = {8241, 318, 1757, 37470, 30};

constexpr size_t NUM_TOKENS_TO_GENERATE = 64;

TEST_F(CompiledModelTest, ResetStateGPT2) {
    // collect reference response tokens
    ov::InferRequest lm = model.create_infer_request();
    std::vector<float> logits_sun_ref = infer_and_get_last_logits(lm, GPT2_SUN_PROMPT_TOKEN_IDS, 0);
    std::vector<int64_t> out_token_ids_ref = generate_n_tokens_with_positions(lm,
                                                                              get_token_from_logits(logits_sun_ref),
                                                                              NUM_TOKENS_TO_GENERATE,
                                                                              GPT2_SUN_PROMPT_TOKEN_IDS.size());

    // call SetUp to reload the model from scratch, process unrelated prompt, then reset and request generation with the
    // first prompt again
    SetUp();

    ov::InferRequest lm_reset = model.create_infer_request();
    std::vector<float> logits_lennon_reset = infer_and_get_last_logits(lm, GPT2_LENNON_PROMPT_TOKEN_IDS, 0);

    lm_reset.reset_state();

    std::vector<float> logits_sun_reset = infer_and_get_last_logits(lm_reset,
                                                                    GPT2_SUN_PROMPT_TOKEN_IDS,
                                                                    0);  // GPT2_LENNON_PROMPT_TOKEN_IDS.size());

    std::vector<int64_t> out_token_ids_reset = generate_n_tokens_with_positions(lm_reset,
                                                                                get_token_from_logits(logits_sun_reset),
                                                                                NUM_TOKENS_TO_GENERATE,
                                                                                GPT2_SUN_PROMPT_TOKEN_IDS.size());
    ASSERT_EQ(out_token_ids_reset, out_token_ids_ref);

    // sanity check - without reset the response after the second prompt is different
    SetUp();

    ov::InferRequest lm_bad = model.create_infer_request();
    std::vector<float> logits_lennon_bad = infer_and_get_last_logits(lm, GPT2_LENNON_PROMPT_TOKEN_IDS, 0);

    // no reset_state on purpose

    std::vector<float> logits_sun_bad = infer_and_get_last_logits(lm_reset,
                                                                  GPT2_SUN_PROMPT_TOKEN_IDS,
                                                                  0);  // GPT2_LENNON_PROMPT_TOKEN_IDS.size());

    std::vector<int64_t> out_token_ids_bad = generate_n_tokens_with_positions(lm_reset,
                                                                              get_token_from_logits(logits_sun_reset),
                                                                              NUM_TOKENS_TO_GENERATE,
                                                                              GPT2_SUN_PROMPT_TOKEN_IDS.size());
}
