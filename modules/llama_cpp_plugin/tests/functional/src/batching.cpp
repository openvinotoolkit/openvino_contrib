// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "llm_inference.hpp"

const std::string MODEL_FILE = ov::test::utils::getCurrentWorkingDir() + SEP + TEST_FILES_DIR + SEP + "gpt2.gguf";

class LlamaCppBatchingDimensionTest : public testing::TestWithParam<ov::Shape> {};

TEST_P(LlamaCppBatchingDimensionTest, BatchedOutputDimensionIsAlignedWithInputDimenstion) {
    ov::Core core;
    auto model = core.compile_model(MODEL_FILE, "LLAMA_CPP");
    auto infer_request = model.create_infer_request();

    auto batched_shape = GetParam();

    auto input_tensor = ov::Tensor(ov::element::Type_t::i64, batched_shape);
    std::fill(input_tensor.data<int64_t>(), input_tensor.data<int64_t>() + input_tensor.get_size(), 0);
    infer_request.set_tensor("input_ids", input_tensor);
    infer_request.set_tensor("position_ids", input_tensor);
    infer_request.infer();
    auto output_shape = infer_request.get_tensor("logits").get_shape();
    ASSERT_EQ(output_shape.size(), 3);  // (batch, input token, output logit distribution)
    auto output_shape_without_logit_dimension = ov::Shape{output_shape[0], output_shape[1]};
    ASSERT_EQ(batched_shape, output_shape_without_logit_dimension);
}

INSTANTIATE_TEST_SUITE_P(VariousBatchAndInputShapes,
                         LlamaCppBatchingDimensionTest,
                         ::testing::Values(ov::Shape{2, 1}, ov::Shape{3, 12}, ov::Shape{13, 7}));

TEST(LlamaCppBatchingTest, BatchedResultIsIdenticalToSingleBatchResults) {
    ov::Core core;
    auto model = core.compile_model(MODEL_FILE, "LLAMA_CPP");
    auto infer_request = model.create_infer_request();

    std::vector<int64_t> mock_input_1{4, 8, 15, 16, 23, 42};
    std::vector<int64_t> mock_input_2{1, 1, 2, 3, 5, 8};

    ASSERT_EQ(mock_input_1.size(), mock_input_2.size());

    infer_request = infer_logits_for_tokens_with_positions(infer_request, mock_input_1, 0);
    auto unbatched_output_1_tensor = infer_request.get_tensor("logits");
    size_t vocab_size = unbatched_output_1_tensor.get_shape().back();

    auto unbatched_output_1 =
        std::vector<float>(unbatched_output_1_tensor.data<float>(),
                           unbatched_output_1_tensor.data<float>() + mock_input_1.size() * vocab_size);

    infer_request.reset_state();

    infer_request = infer_logits_for_tokens_with_positions(infer_request, mock_input_2, 0);
    auto unbatched_output_2_tensor = infer_request.get_tensor("logits");
    auto unbatched_output_2 =
        std::vector<float>(unbatched_output_2_tensor.data<float>(),
                           unbatched_output_2_tensor.data<float>() + mock_input_2.size() * vocab_size);
    infer_request.reset_state();

    auto batched_input_ids = ov::Tensor(ov::element::Type_t::i64, ov::Shape{2, mock_input_1.size()});
    size_t midpoint_offset = mock_input_1.size();
    auto end_offset = midpoint_offset * 2;

    std::copy(mock_input_1.begin(), mock_input_1.end(), batched_input_ids.data<int64_t>());
    std::copy(mock_input_2.begin(), mock_input_2.end(), batched_input_ids.data<int64_t>() + midpoint_offset);
    infer_request.set_tensor("input_ids", batched_input_ids);

    auto batched_position_ids = ov::Tensor(ov::element::Type_t::i64, ov::Shape{2, mock_input_1.size()});
    std::iota(batched_position_ids.data<int64_t>(), batched_position_ids.data<int64_t>() + midpoint_offset, 0);
    std::iota(batched_position_ids.data<int64_t>() + midpoint_offset,
              batched_position_ids.data<int64_t>() + end_offset,
              0);
    infer_request.set_tensor("position_ids", batched_position_ids);
    infer_request.infer();

    auto batched_output = infer_request.get_tensor("logits");
    auto batched_output_1 =
        std::vector<float>(batched_output.data<float>(), batched_output.data<float>() + midpoint_offset * vocab_size);
    auto batched_output_2 = std::vector<float>(batched_output.data<float>() + midpoint_offset * vocab_size,
                                               batched_output.data<float>() + end_offset * vocab_size);

    EXPECT_EQ(unbatched_output_1.size(), batched_output_1.size());
    EXPECT_EQ(unbatched_output_2.size(), batched_output_2.size());

    EXPECT_EQ(unbatched_output_1, batched_output_1);
    EXPECT_EQ(unbatched_output_2, batched_output_2);
}
