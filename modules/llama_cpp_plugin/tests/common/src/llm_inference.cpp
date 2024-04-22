#include "llm_inference.hpp"

ov::InferRequest& infer_logits_for_tokens_with_positions(ov::InferRequest& infer_request,
                                                         const std::vector<int64_t>& tokens,
                                                         int64_t position_ids_start_value) {
    auto input_ids_tensor = ov::Tensor(ov::element::Type_t::i64, {1, tokens.size()});
    std::copy(tokens.begin(), tokens.end(), input_ids_tensor.data<int64_t>());
    infer_request.set_tensor("input_ids", input_ids_tensor);

    ov::Tensor position_ids = infer_request.get_tensor("position_ids");
    position_ids.set_shape(input_ids_tensor.get_shape());
    std::iota(position_ids.data<int64_t>(),
              position_ids.data<int64_t>() + position_ids.get_size(),
              position_ids_start_value);

    CompiledModelTest::fill_unused_inputs(infer_request, input_ids_tensor.get_shape());
    infer_request.infer();
    return infer_request;
}

// Infers all tokens, but returns only the logits for the last token in `tokens`.
std::vector<float> infer_and_get_last_logits(ov::InferRequest& infer_request,
                                             const std::vector<int64_t>& tokens,
                                             int64_t position_ids_start_value) {
    infer_request = infer_logits_for_tokens_with_positions(infer_request, tokens, position_ids_start_value);
    size_t vocab_size = infer_request.get_tensor("logits").get_shape().back();
    float* logits = infer_request.get_tensor("logits").data<float>() + (tokens.size() - 1) * vocab_size;
    std::vector<float> logits_vector(vocab_size);
    std::copy(logits, logits + vocab_size, logits_vector.begin());
    return logits_vector;
}

std::vector<int64_t> generate_n_tokens_with_positions(ov::InferRequest& infer_request,
                                                      int64_t last_token,
                                                      size_t n_tokens,
                                                      int64_t position_ids_start_value) {
    size_t cnt = 0;
    std::vector<int64_t> out_token_ids;
    out_token_ids.push_back(last_token);

    while (cnt < n_tokens) {
        std::vector<float> logits_curr =
            infer_and_get_last_logits(infer_request, {out_token_ids.back()}, cnt + position_ids_start_value);
        int64_t out_token = std::max_element(logits_curr.begin(), logits_curr.end()) - logits_curr.begin();
        out_token_ids.push_back(out_token);
        cnt++;
    }
    return out_token_ids;
}
