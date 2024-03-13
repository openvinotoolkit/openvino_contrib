#include <gtest/gtest.h>

#include "common_test_utils/file_utils.hpp"
#include "openvino/openvino.hpp"

const std::string TEST_FILES_DIR = "test_data";

// "Why is the Sun yellow?"
const std::vector<int64_t> GPT2_PROMPT_TOKEN_IDS = {5195, 318, 262, 3825, 7872, 30};
// "The Sun is a bright red, which means it is a bright red. The Sun is a bright red because it is a bright red."
const std::vector<int64_t> GPT2_REFERENCE_RESPONSE_TOKEN_IDS = {
    198, 464,  3825, 318, 257,  6016, 2266, 11,  543, 1724, 340,  318,  257, 6016, 2266, 13,
    383, 3825, 318,  257, 6016, 2266, 780,  340, 318, 257,  6016, 2266, 13,  198,  198,  464};

const auto SEP = ov::util::FileTraits<char>::file_separator;

TEST(PromptResponseTest, TestGPT2) {
    const std::string plugin_name = "LLAMA_CPP";
    ov::Core core;

    const std::string model_file_name = "gpt2.gguf";
    const std::string model_file =
        ov::test::utils::getCurrentWorkingDir() + SEP + TEST_FILES_DIR + SEP + model_file_name;
    ov::InferRequest lm = core.compile_model(model_file, plugin_name).create_infer_request();
    auto input_ids_tensor = ov::Tensor(ov::element::Type_t::i64, {1, GPT2_PROMPT_TOKEN_IDS.size()});
    std::copy(GPT2_PROMPT_TOKEN_IDS.begin(), GPT2_PROMPT_TOKEN_IDS.end(), input_ids_tensor.data<int64_t>());
    lm.set_tensor("input_ids", input_ids_tensor);
    lm.set_tensor("attention_mask", ov::Tensor(ov::element::Type_t::i64, {1, GPT2_PROMPT_TOKEN_IDS.size()}));
    ov::Tensor position_ids = lm.get_tensor("position_ids");
    position_ids.set_shape(input_ids_tensor.get_shape());
    std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + position_ids.get_size(), 0);

    constexpr size_t BATCH_SIZE = 1;
    lm.get_tensor("beam_idx").set_shape({BATCH_SIZE});
    lm.get_tensor("beam_idx").data<int32_t>()[0] = 0;

    lm.infer();

    size_t vocab_size = lm.get_tensor("logits").get_shape().back();
    float* logits = lm.get_tensor("logits").data<float>() + (input_ids_tensor.get_size() - 1) * vocab_size;
    int64_t out_token = std::max_element(logits, logits + vocab_size) - logits;

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
