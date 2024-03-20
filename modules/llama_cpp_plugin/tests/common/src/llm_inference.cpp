#include "llm_inference.hpp"

std::vector<float> infer_logits_for_tokens_with_positions(ov::InferRequest& lm,
                                                          const std::vector<int64_t>& tokens,
                                                          int64_t position_ids_start_value) {
    auto input_ids_tensor = ov::Tensor(ov::element::Type_t::i64, {1, tokens.size()});
    std::copy(tokens.begin(), tokens.end(), input_ids_tensor.data<int64_t>());
    lm.set_tensor("input_ids", input_ids_tensor);

    ov::Tensor position_ids = lm.get_tensor("position_ids");
    position_ids.set_shape(input_ids_tensor.get_shape());
    std::iota(position_ids.data<int64_t>(),
              position_ids.data<int64_t>() + position_ids.get_size(),
              position_ids_start_value);

    CompiledModelTest::fill_unused_inputs(lm, input_ids_tensor.get_shape());
    lm.infer();

    size_t vocab_size = lm.get_tensor("logits").get_shape().back();
    float* logits = lm.get_tensor("logits").data<float>() + (input_ids_tensor.get_size() - 1) * vocab_size;
    std::vector<float> logits_vector(vocab_size);
    std::copy(logits, logits + vocab_size, logits_vector.begin());
    return logits_vector;
}

std::vector<int64_t> generate_n_tokens_with_positions(ov::InferRequest& lm,
                                                      int64_t last_token,
                                                      size_t n_tokens,
                                                      int64_t position_ids_start_value) {
    size_t cnt = 0;
    std::vector<int64_t> out_token_ids;
    out_token_ids.push_back(last_token);

    while (cnt < n_tokens) {
        std::vector<float> logits_curr =
            infer_logits_for_tokens_with_positions(lm, {out_token_ids.back()}, cnt + position_ids_start_value);
        int64_t out_token = std::max_element(logits_curr.begin(), logits_curr.end()) - logits_curr.begin();
        out_token_ids.push_back(out_token);
        cnt++;
    }
    return out_token_ids;
}
