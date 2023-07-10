// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "normalizer.h" // for absl::string_view

#include "openvino/op/util/framework_node.hpp"
#include "openvino/opsets/opset10.hpp"

#include "fast_tokenizer/normalizers/normalizers.h"
#include "fast_tokenizer/models/models.h"
#include "fast_tokenizer/pretokenizers/pretokenizers.h"

#include "regex_split.hpp"
#include "utils.hpp"

using namespace ov;


namespace {

using paddlenlp::fast_tokenizer::core::SplitMode;
const std::map<std::string, SplitMode> split_modes = {
    {"remove", SplitMode::REMOVED},
    {"isolate", SplitMode::ISOLATED},
    {"contiguous", SplitMode::CONTIGUOUS},
    {"merge_with_previous", SplitMode::MERGED_WITH_PREVIOUS},
    {"merge_with_next", SplitMode::MERGED_WITH_NEXT},
};

}


void RegexSplit::validate_and_infer_types() {
//    check_string_input(this, 0);
//    check_string_scalar_input(this, 3);
//    check_ragged_string_input(this, 0);
//    check_string_input(this, 5);
    OPENVINO_ASSERT(split_modes.find(m_behaviour) != split_modes.end(), "RegexSplit doesn't support unknown split mode: " + m_behaviour);
    set_ragged_string_output(this, 0, get_input_partial_shape(0));
}

bool RegexSplit::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {

    if (inputs.size() < 5) {
        auto begins = inputs[0].data<const int32_t>();
        auto ends   = inputs[1].data<const int32_t>();
        auto chars  = inputs[2].data<const uint8_t>();

        ov::Tensor ragged_begins_tensor(ov::element::i32, inputs[0].get_shape());
        ov::Tensor ragged_ends_tensor(ov::element::i32, inputs[0].get_shape());
        auto ragged_begins = ragged_begins_tensor.data<int32_t>();
        auto ragged_ends = ragged_ends_tensor.data<int32_t>();
        for (int i=0; i < inputs[0].get_size(); ++i) {
            ragged_begins[i] = i;
            ragged_ends[i] = i + 1;
        };

        auto split_pattern_buf  = inputs[3].data<const uint8_t>();
        auto split_pattern = absl::string_view((const char*)split_pattern_buf, shape_size(inputs[3].get_shape()) - 1);   // Shouldn't be applied FIXME: -1 is a complementary change to a WA applied in string_attribute_to_constant

//        std::cerr << "[ RegexSplit ] Split Pattern: `" << split_pattern << "`, behaviour: " << m_behaviour << "\n";

        const size_t num_rows = inputs[0].get_size();
        const size_t num_chars = inputs[2].get_size();

        outputs[0].set_shape(inputs[0].get_shape());
        outputs[1].set_shape(inputs[1].get_shape());

        outputs[2].set_shape(Shape{num_chars});
        outputs[3].set_shape(Shape{num_chars});

        outputs[4] = inputs[2];

        // For the whole implementation below the input shapes can be ignored, we are working with the flatten representaions
        // and only number of elements in the original tensors matter

        // Get pointers in the output tensors
        auto new_ragged_begins = outputs[0].data<int32_t>();
        auto new_ragged_ends   = outputs[1].data<int32_t>();
        auto new_begins = outputs[2].data<int32_t>();
        auto new_ends   = outputs[3].data<int32_t>();
        int32_t ragged_offset = 0;

        using namespace paddlenlp::fast_tokenizer;
        auto pretokenizer = pretokenizers::SplitPreTokenizer(std::string(split_pattern), split_modes.at(m_behaviour), m_invert);

        for(size_t seq = 0; seq < num_rows; ++seq) {
//            std::cerr << "================= Seq: " << seq << " ====================\n";
//            std::cerr << "Ragged begins: " << ragged_begins[seq] << "; Ragged Ends: " << ragged_ends[seq] << "\n";

            new_ragged_begins[seq] = ragged_offset;

            for(size_t ragged_col = ragged_begins[seq]; ragged_col < ragged_ends[seq]; ++ragged_col) {

                auto str = std::string(chars + begins[ragged_col], chars + ends[ragged_col]);
//                std::cerr << "[ RegexSplit ] old_str: '" << str << "'\n";
                paddlenlp::fast_tokenizer::pretokenizers::PreTokenizedString pretokenized(str);
                pretokenizer(&pretokenized);
                size_t num_splits = pretokenized.GetSplitsSize();
//                std::cerr << "[ RegexSplit ] num_splits: " << num_splits << "\n";

                for (size_t j = 0; j < num_splits; ++j) {
                    auto split = pretokenized.GetSplit(j);
                    const auto& value = split.normalized_.GetStr();
                    auto offset = split.normalized_.GetOrginalOffset();
//                    std::cerr << "[ RegexSplit ]     split part: '" << value << "'\n";
//                    std::cerr << "[ RegexSplit ]     split offs: " << offset.first << ":" << offset.second << "\n";
                    new_begins[ragged_offset] = begins[ragged_col] + offset.first;
                    new_ends[ragged_offset++] = begins[ragged_col] + offset.second;
                };
            }

            new_ragged_ends[seq] = ragged_offset;
        }

        // Fix real shape based on collected results
        outputs[2].set_shape({size_t(ragged_offset)});
        outputs[3].set_shape({size_t(ragged_offset)});

    } else {
        auto ragged_begins = inputs[0].data<const int32_t>();
        auto ragged_ends   = inputs[1].data<const int32_t>();
        auto begins = inputs[2].data<const int32_t>();
        auto ends   = inputs[3].data<const int32_t>();
        auto chars  = inputs[4].data<const uint8_t>();

        auto split_pattern_buf  = inputs[5].data<const uint8_t>();
        auto split_pattern = absl::string_view((const char*)split_pattern_buf, shape_size(inputs[5].get_shape())/* - 1*/);   // Shouldn't be applied FIXME: -1 is a complementary change to a WA applied in string_attribute_to_constant

//        std::cerr << "Split Pattern: `" << split_pattern << "`, behaviour: " << m_behaviour << "\n";

        outputs[4] = inputs[4];
        const size_t num_rows = inputs[0].get_size();
        const size_t num_chars = inputs[4].get_size();

        outputs[0].set_shape(inputs[0].get_shape());
        outputs[1].set_shape(inputs[1].get_shape());

        outputs[2].set_shape(Shape{num_chars});
        outputs[3].set_shape(Shape{num_chars});

        outputs[4] = inputs[4];

        // For the whole implementation below the input shapes can be ignored, we are working with the flatten representaions
        // and only number of elements in the original tensors matter

        // Get pointers in the output tensors
        auto new_ragged_begins = outputs[0].data<int32_t>();
        auto new_ragged_ends   = outputs[1].data<int32_t>();
        auto new_begins = outputs[2].data<int32_t>();
        auto new_ends   = outputs[3].data<int32_t>();
        int32_t ragged_offset = 0;

        using namespace paddlenlp::fast_tokenizer;
        auto pretokenizer = pretokenizers::SplitPreTokenizer(std::string(split_pattern), split_modes.at(m_behaviour), m_invert);

        for(size_t seq = 0; seq < num_rows; ++seq) {
//            std::cerr << "----------------- Seq: " << seq << " -----------------\n";
//            std::cerr << "Ragged begins: " << ragged_begins[seq] << "; Ragged Ends: " << ragged_ends[seq] << "\n";

            new_ragged_begins[seq] = ragged_offset;

            for(size_t ragged_col = ragged_begins[seq]; ragged_col < ragged_ends[seq]; ++ragged_col) {
                auto str = std::string(chars + begins[ragged_col], chars + ends[ragged_col]);
//                std::cerr << "[ RegexSplit ] old_str: '" << str << "'\n";
                paddlenlp::fast_tokenizer::pretokenizers::PreTokenizedString pretokenized(str);
                pretokenizer(&pretokenized);
                size_t num_splits = pretokenized.GetSplitsSize();


                for (size_t j = 0; j < num_splits; ++j) {
                    auto split = pretokenized.GetSplit(j);
                    const auto& value = split.normalized_.GetStr();
                    auto offset = split.normalized_.GetOrginalOffset();
//                    std::cerr << "[ RegexSplit ]     split part: '" << value << "'\n";
//                    std::cerr << "[ RegexSplit ]     split offs: " << offset.first << ":" << offset.second << "\n";
                    new_begins[ragged_offset] = begins[ragged_col] + offset.first;
                    new_ends[ragged_offset++] = begins[ragged_col] + offset.second;


//                    std::cerr << "New begins and ends:\n";
//                    for (size_t i = 0; i < outputs[2].get_size(); ++i) {
//                        std::cerr << outputs[2].data<int32_t>()[i] << ", ";
//                    }
//                    std::cerr << "\n";
//
//                    for (size_t i = 0; i < outputs[3].get_size(); ++i) {
//                        std::cerr << outputs[3].data<int32_t>()[i] << ", ";
//                    }
//                    std::cerr << "\n";
                };
            }

            new_ragged_ends[seq] = ragged_offset;
        }

        // Fix real shape based on collected results
        outputs[2].set_shape({size_t(ragged_offset)});
        outputs[3].set_shape({size_t(ragged_offset)});

    }

    return true;
}
