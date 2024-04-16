// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifndef LLM_INFERENCE_HPP
#define LLM_INFERENCE_HPP

#include "model_fixture.hpp"
#include "openvino/openvino.hpp"

ov::InferRequest& infer_logits_for_tokens_with_positions(ov::InferRequest& infer_request,
                                                         const std::vector<int64_t>& tokens,
                                                         int64_t position_ids_start_value);
std::vector<float> infer_and_get_last_logits(ov::InferRequest& lm,
                                             const std::vector<int64_t>& tokens,
                                             int64_t position_ids_start_value);

std::vector<int64_t> generate_n_tokens_with_positions(ov::InferRequest& lm,
                                                      int64_t last_token,
                                                      size_t n_tokens,
                                                      int64_t position_ids_start_value);

inline int64_t get_token_from_logits(const std::vector<float>& logits) {
    return std::max_element(logits.cbegin(), logits.cend()) - logits.cbegin();
}
#endif /* LLM_INFERENCE_HPP */
