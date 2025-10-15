// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_finite_comparer.hpp"

using namespace ov::test;

void ov::test::FiniteLayerComparer::compare(const std::vector<ov::Tensor>& expected_outputs,
                                            const std::vector<ov::Tensor>& actual_outputs,
                                            float threshold,
                                            bool to_check_nans,
                                            std::optional<double> infinity_value) {
    for (std::size_t output_iIndex = 0; output_iIndex < expected_outputs.size(); ++output_iIndex) {
        const auto& expected = expected_outputs[output_iIndex];
        const auto& actual = actual_outputs[output_iIndex];
        FiniteLayerComparer::compare(expected, actual, threshold, to_check_nans, infinity_value);
    }
}

template <typename T_IE>
inline void call_compare(const ov::Tensor& expected,
                         const T_IE* actual_buffer,
                         size_t size,
                         float threshold,
                         bool to_check_nans,
                         std::optional<double> infinity_value) {
    const auto& precision = expected.get_element_type();
    switch (precision) {
        case ov::element::Type_t::i64:
            FiniteLayerComparer::compare<T_IE>(
                expected.data<int64_t>(), actual_buffer, size, threshold, to_check_nans, infinity_value);
            break;
        case ov::element::Type_t::i32:
            FiniteLayerComparer::compare<T_IE>(
                expected.data<int32_t>(), actual_buffer, size, threshold, to_check_nans, infinity_value);
            break;
        case ov::element::Type_t::i16:
            FiniteLayerComparer::compare<T_IE>(
                expected.data<int16_t>(), actual_buffer, size, threshold, to_check_nans, infinity_value);
            break;
        case ov::element::Type_t::i8:
            FiniteLayerComparer::compare<T_IE>(
                expected.data<int8_t>(), actual_buffer, size, threshold, to_check_nans, infinity_value);
            break;
        case ov::element::Type_t::u64:
            FiniteLayerComparer::compare<T_IE>(
                expected.data<uint64_t>(), actual_buffer, size, threshold, to_check_nans, infinity_value);
            break;
        case ov::element::Type_t::u32:
            FiniteLayerComparer::compare<T_IE>(
                expected.data<uint32_t>(), actual_buffer, size, threshold, to_check_nans, infinity_value);
            break;
        case ov::element::Type_t::u16:
            FiniteLayerComparer::compare<T_IE>(
                expected.data<uint16_t>(), actual_buffer, size, threshold, to_check_nans, infinity_value);
            break;
        case ov::element::Type_t::boolean:
        case ov::element::Type_t::u8:
            FiniteLayerComparer::compare<T_IE>(
                expected.data<uint8_t>(), actual_buffer, size, threshold, to_check_nans, infinity_value);
            break;
        case ov::element::Type_t::f64:
            FiniteLayerComparer::compare<T_IE>(
                expected.data<double>(), actual_buffer, size, threshold, to_check_nans, infinity_value);
            break;
        case ov::element::Type_t::f32:
            FiniteLayerComparer::compare<T_IE>(
                expected.data<float>(), actual_buffer, size, threshold, to_check_nans, infinity_value);
            break;
        case ov::element::Type_t::f16:
            FiniteLayerComparer::compare<T_IE>(
                expected.data<ov::float16>(), actual_buffer, size, threshold, to_check_nans, infinity_value);
            break;
        case ov::element::Type_t::bf16:
            FiniteLayerComparer::compare<T_IE>(
                expected.data<ov::bfloat16>(), actual_buffer, size, threshold, to_check_nans, infinity_value);
            break;
        case ov::element::Type_t::dynamic:
            FiniteLayerComparer::compare<T_IE, T_IE>(
                expected.data<T_IE>(), actual_buffer, size, threshold, to_check_nans, infinity_value);
            break;
        default:
            FAIL() << "Comparator for " << precision << " precision isn't supported";
    }
    return;
}

void FiniteLayerComparer::compare(const ov::Tensor& expected,
                                  const ov::Tensor& actual,
                                  float threshold,
                                  bool to_check_nans,
                                  std::optional<double> infinity_value) {
    const auto& precision = actual.get_element_type();
    auto k = static_cast<float>(expected.get_element_type().size()) / precision.size();
    // W/A for int4, uint4
    if (expected.get_element_type() == ov::element::Type_t::u4 ||
        expected.get_element_type() == ov::element::Type_t::i4) {
        k /= 2;
    } else if (expected.get_element_type() == ov::element::Type_t::dynamic) {
        k = 1;
    }
    ASSERT_EQ(expected.get_byte_size(), actual.get_byte_size() * k);

    const auto& size = actual.get_size();
    switch (precision) {
        case ov::element::f32:
            call_compare(expected, actual.data<float>(), size, threshold, to_check_nans, infinity_value);
            break;
        case ov::element::i32:
            call_compare(expected, actual.data<int32_t>(), size, threshold, to_check_nans, infinity_value);
            break;
        case ov::element::u32:
            call_compare(expected, actual.data<uint32_t>(), size, threshold, to_check_nans, infinity_value);
            break;
        case ov::element::i64:
            call_compare(expected, actual.data<int64_t>(), size, threshold, to_check_nans, infinity_value);
            break;
        case ov::element::i8:
            call_compare(expected, actual.data<int8_t>(), size, threshold, to_check_nans, infinity_value);
            break;
        case ov::element::u16:
            call_compare(expected, actual.data<uint16_t>(), size, threshold, to_check_nans, infinity_value);
            break;
        case ov::element::i16:
            call_compare(expected, actual.data<int16_t>(), size, threshold, to_check_nans, infinity_value);
            break;
        case ov::element::boolean:
        case ov::element::u8:
            call_compare(expected, actual.data<uint8_t>(), size, threshold, to_check_nans, infinity_value);
            break;
        case ov::element::u64:
            call_compare(expected, actual.data<uint64_t>(), size, threshold, to_check_nans, infinity_value);
            break;
        case ov::element::bf16:
            call_compare(expected, actual.data<ov::bfloat16>(), size, threshold, to_check_nans, infinity_value);
            break;
        case ov::element::f16:
            call_compare(expected, actual.data<ov::float16>(), size, threshold, to_check_nans, infinity_value);
            break;
        default:
            FAIL() << "Comparator for " << precision << " precision isn't supported";
    }
}

void ov::test::FiniteLayerComparer::compare(const std::vector<ov::Tensor>& expected_outputs,
                                            const std::vector<ov::Tensor>& actual_outputs) {
    FiniteLayerComparer::compare(
        expected_outputs, actual_outputs, abs_threshold, this->to_check_nans, this->infinity_value);
}
