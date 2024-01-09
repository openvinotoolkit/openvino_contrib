// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef _MSC_VER
#    pragma warning(disable : 4251)
#    pragma warning(disable : 4275)
#endif

#include "fast_tokenizer/normalizers/normalizers.h"

#include "normalize_unicode.hpp"
#include "utils.hpp"

using namespace ov;

namespace {
using namespace paddlenlp::fast_tokenizer::normalizers;
using NormalizersMap = std::map<std::string, std::function<std::string(const std::string&)>>;

const NormalizersMap normalizers = {
    {"NFD", [](const std::string& str) { return NormalizedString(str).NFD().GetStr(); }},
    {"NFC", [](const std::string& str) { return NormalizedString(str).NFC().GetStr(); }},
    {"NFKD", [](const std::string& str) { return NormalizedString(str).NFKD().GetStr(); }},
    {"NFKC", [](const std::string& str) { return NormalizedString(str).NFKC().GetStr(); }},
};

}

void NormalizeUnicode::validate_and_infer_types() {
    check_string_input(this, 0);
    OPENVINO_ASSERT(normalizers.find(m_normalization_form) != normalizers.end(), "NormalizeUnicode doesn't know normalization form " + m_normalization_form);
    set_string_output(this, 0, get_input_partial_shape(0));
}

bool NormalizeUnicode::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    return evaluate_normalization_helper(outputs, inputs, normalizers.at(m_normalization_form));
}
