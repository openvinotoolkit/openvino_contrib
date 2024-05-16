// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef MODEL_FIXTURE_HPP
#define MODEL_FIXTURE_HPP

#include <gtest/gtest.h>

#include "common_test_utils/file_utils.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/infer_request.hpp"

const std::string TEST_FILES_DIR = "test_data";
const auto SEP = ov::util::FileTraits<char>::file_separator;

class CompiledModelTest : public ::testing::Test {
public:
    static void fill_unused_inputs(ov::InferRequest& infer_request, const ov::Shape& input_ids_reference_shape) {
        infer_request.set_tensor("attention_mask", ov::Tensor(ov::element::Type_t::i64, input_ids_reference_shape));

        size_t batch_size = input_ids_reference_shape[0];
        infer_request.set_tensor("beam_idx", ov::Tensor(ov::element::Type_t::i32, ov::Shape{batch_size}));
    }

protected:
    void SetUp() override {
        const std::string plugin_name = "LLAMA_CPP";
        ov::Core core;

        const std::string model_file_name = "gpt2.gguf";
        const std::string model_file =
            ov::test::utils::getCurrentWorkingDir() + SEP + TEST_FILES_DIR + SEP + model_file_name;
        model = core.compile_model(model_file, plugin_name);
    }
    ov::CompiledModel model;
};

#endif /* MODEL_FIXTURE_HPP */
