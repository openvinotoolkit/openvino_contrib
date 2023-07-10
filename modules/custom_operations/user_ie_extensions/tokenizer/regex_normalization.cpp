// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "normalizer.h" // for absl::string_view

#include "fast_tokenizer/normalizers/normalizers.h"
#include "fast_tokenizer/models/models.h"
#include "fast_tokenizer/pretokenizers/pretokenizers.h"

#include "regex_normalization.hpp"
#include "utils.hpp"

using namespace ov;


void RegexNormalization::validate_and_infer_types() {
    check_string_input(this, 0);
    check_string_scalar_input(this, 3);
    check_string_scalar_input(this, 4);
    set_string_output(this, 0, get_input_partial_shape(0));
}

bool RegexNormalization::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto search_pattern_buf  = inputs[3].data<const uint8_t>();
    auto replace_pattern_buf  = inputs[4].data<const uint8_t>();
    auto search_pattern = absl::string_view((const char*)search_pattern_buf, shape_size(inputs[3].get_shape()) - 1);   // FIXME: -1 is a complementary change to a WA applied in string_attribute_to_constant
    auto replace_pattern = absl::string_view((const char*)replace_pattern_buf, shape_size(inputs[4].get_shape()) - 1);   // FIXME: -1 is a complementary change to a WA applied in string_attribute_to_constant

    using namespace paddlenlp::fast_tokenizer::normalizers;
    re2::RE2 search_pattern_re(search_pattern);

    return evaluate_normalization_helper(
        outputs, inputs,
        [&replace_pattern, &search_pattern_re](const std::string& str) {
            return NormalizedString(str).Replace(search_pattern_re, std::string(replace_pattern)).GetStr();
    });
}
