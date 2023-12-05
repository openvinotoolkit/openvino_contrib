// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "case_fold.hpp"
#include "utils.hpp"

#include "fast_tokenizer/normalizers/normalizers.h"

using namespace ov;


void CaseFold::validate_and_infer_types() {
    check_string_input(this, 0);
    set_string_output(this, 0, get_input_partial_shape(0));
}

bool CaseFold::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    return evaluate_normalization_helper(
        outputs, inputs,
        [](const std::string& str) {
            using namespace paddlenlp::fast_tokenizer;
            return normalizers::NormalizedString(str).Lowercase().GetStr();
        });
}
