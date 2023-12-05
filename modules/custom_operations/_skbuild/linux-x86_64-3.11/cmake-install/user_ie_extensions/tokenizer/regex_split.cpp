// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/framework_node.hpp"
#include "openvino/opsets/opset10.hpp"

#include "regex_split.hpp"
#include "utils.hpp"
#include "fast_tokenizer/normalizers/normalizers.h"

using namespace ov;
using namespace ov::opset10;

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


RegexSplit::RegexSplit(const ov::OutputVector& arguments, const std::string& behaviour, bool invert) :
    ov::op::Op(arguments),
    m_behaviour(behaviour),
    m_invert(invert) {
    constructor_validate_and_infer_types();
}


RegexSplit::RegexSplit(
    const ov::OutputVector& arguments,
    const std::shared_ptr<pretokenizers::SplitPreTokenizer>& pretokenizer,
    const std::string& behaviour,
    bool invert
) :
    ov::op::Op(arguments),
    m_pretokenizer(pretokenizer),
    m_behaviour(behaviour),
    m_invert(invert) {

    if (m_pretokenizer == nullptr) {
        auto split_pattern_const = as_type_ptr<Constant>(arguments[5].get_node_shared_ptr());
        auto split_pattern_buf = static_cast<const char*>(split_pattern_const->get_data_ptr());
        auto split_pattern = std::string(split_pattern_buf, split_pattern_const->get_byte_size());
        m_pretokenizer = std::make_shared<pretokenizers::SplitPreTokenizer>(split_pattern, split_modes.at(behaviour), invert);
    };

    constructor_validate_and_infer_types();
}


void RegexSplit::validate_and_infer_types() {
    check_ragged_string_input(this, 0);
    check_string_scalar_input(this, 5);
    OPENVINO_ASSERT(split_modes.find(m_behaviour) != split_modes.end(), "RegexSplit doesn't support unknown split mode: " + m_behaviour);
    set_ragged_string_output(this, 0, get_input_partial_shape(0));
}

bool RegexSplit::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto ragged_begins = inputs[0].data<const int32_t>();
    auto ragged_ends   = inputs[1].data<const int32_t>();
    auto begins = inputs[2].data<const int32_t>();
    auto ends   = inputs[3].data<const int32_t>();
    auto chars  = inputs[4].data<const uint8_t>();

    outputs[4] = inputs[4];
    const size_t num_rows = inputs[0].get_size();
    const size_t num_chars = inputs[4].get_size();

    outputs[0].set_shape(inputs[0].get_shape());
    outputs[1].set_shape(inputs[1].get_shape());

    outputs[2].set_shape(Shape{num_chars});
    outputs[3].set_shape(Shape{num_chars});

    outputs[4] = inputs[4];

    // Get pointers in the output tensors
    auto new_ragged_begins = outputs[0].data<int32_t>();
    auto new_ragged_ends   = outputs[1].data<int32_t>();
    auto new_begins = outputs[2].data<int32_t>();
    auto new_ends   = outputs[3].data<int32_t>();
    int32_t ragged_offset = 0;

    for(size_t seq = 0; seq < num_rows; ++seq) {
        new_ragged_begins[seq] = ragged_offset;

        for(size_t ragged_col = ragged_begins[seq]; ragged_col < ragged_ends[seq]; ++ragged_col) {
            auto str = std::string(chars + begins[ragged_col], chars + ends[ragged_col]);
            paddlenlp::fast_tokenizer::pretokenizers::PreTokenizedString pretokenized(str);
            (*m_pretokenizer)(&pretokenized);
            size_t num_splits = pretokenized.GetSplitsSize();

            for (size_t j = 0; j < num_splits; ++j) {
                auto split = pretokenized.GetSplit(j);
                const auto& value = split.normalized_.GetStr();
                auto offset = split.normalized_.GetOrginalOffset();
                new_begins[ragged_offset] = begins[ragged_col] + offset.first;
                new_ends[ragged_offset++] = begins[ragged_col] + offset.second;
            };
        }

        new_ragged_ends[seq] = ragged_offset;
    }

    // Fix real shape based on collected results
    outputs[2].set_shape({size_t(ragged_offset)});
    outputs[3].set_shape({size_t(ragged_offset)});

    return true;
}
