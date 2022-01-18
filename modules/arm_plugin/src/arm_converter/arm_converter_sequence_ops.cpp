// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/runtime/reference/reverse_sequence.hpp>
#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/sequences.hpp>

namespace ArmPlugin {

template<> Converter::Conversion::Ptr Converter::Convert(const opset::ReverseSequence& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.output(0),
                                    node.get_input_shape(0),
                                    node.get_batch_axis(),
                                    node.get_sequence_axis(),
                                    node.input(1));
    };
    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::reverse_sequence),
        node.input(0), allTypes,
        node.input(1), indexTypes);
}

template <typename T, typename U>
void wrap_lstm_sequence(const T* X,
                        const ngraph::Shape& X_shape,
                        const T* H,
                        const ngraph::Shape& H_shape,
                        const T* C,
                        const ngraph::Shape& C_shape,
                        const U* seq_lengths,
                        const ngraph::Shape& seq_lengths_shape,
                        const T* W,
                        const ngraph::Shape& W_shape,
                        const T* R,
                        const ngraph::Shape& R_shape,
                        const T* B,
                        const ngraph::Shape& B_shape,
                        T* Y,
                        T* Ho,
                        T* Co,
                        const std::string& activation_f,
                        const std::string& activation_g,
                        const std::string& activation_h,
                        float clip,
                        ngraph::op::RecurrentSequenceDirection direction) {
     ngraph::runtime::reference::lstm_sequence<T, U>(reinterpret_cast<const char*>(X),
                                                    X_shape,
                                                    reinterpret_cast<const char*>(H),
                                                    H_shape,
                                                    reinterpret_cast<const char*>(C),
                                                    C_shape,
                                                    reinterpret_cast<const char*>(seq_lengths),
                                                    seq_lengths_shape,
                                                    reinterpret_cast<const char*>(W),
                                                    W_shape,
                                                    reinterpret_cast<const char*>(R),
                                                    R_shape,
                                                    reinterpret_cast<const char*>(B),
                                                    B_shape,
                                                    reinterpret_cast<char*>(Y),
                                                    reinterpret_cast<char*>(Ho),
                                                    reinterpret_cast<char*>(Co),
                                                    activation_f,
                                                    activation_g,
                                                    activation_h,
                                                    clip,
                                                    direction);
}


template<> Converter::Conversion::Ptr Converter::Convert(const opset::LSTMSequence& node) {
    auto make = [&] (auto refFunction) {
    return this->MakeConversion(refFunction,
                                node.input(0),
                                node.get_input_shape(0),
                                node.input(1),
                                node.get_input_shape(1),
                                node.input(2),
                                node.get_input_shape(2),
                                node.input(3),
                                node.get_input_shape(3),
                                node.input(4),
                                node.get_input_shape(4),
                                node.input(5),
                                node.get_input_shape(5),
                                node.input(6),
                                node.get_input_shape(6),
                                node.output(0),
                                node.output(1),
                                node.output(2),
                                node.get_activations()[0],
                                node.get_activations()[1],
                                node.get_activations()[2],
                                node.get_clip(),
                                node.get_direction());
    };

    return CallSwitch(
        AP_WRAP(make, wrap_lstm_sequence),
        node.input(0), allTypes,
        node.input(3), indexTypes);
}

template <typename T, typename U>
void wrap_gru_sequence(const T* X,
                        const ngraph::Shape& X_shape,
                        const T* H,
                        const ngraph::Shape& H_shape,
                        const U* seq_lengths,
                        const ngraph::Shape& seq_lengths_shape,
                        const T* W,
                        const ngraph::Shape& W_shape,
                        const T* R,
                        const ngraph::Shape& R_shape,
                        const T* B,
                        const ngraph::Shape& B_shape,
                        T* Y,
                        T* Ho,
                        const std::string& activation_f,
                        const std::string& activation_g,
                        const float clip,
                        const ngraph::op::RecurrentSequenceDirection direction,
                        const bool linear_before_reset) {
     ngraph::runtime::reference::gru_sequence<T, U>(reinterpret_cast<const char*>(X),
                                                    X_shape,
                                                    reinterpret_cast<const char*>(H),
                                                    H_shape,
                                                    reinterpret_cast<const char*>(seq_lengths),
                                                    seq_lengths_shape,
                                                    reinterpret_cast<const char*>(W),
                                                    W_shape,
                                                    reinterpret_cast<const char*>(R),
                                                    R_shape,
                                                    reinterpret_cast<const char*>(B),
                                                    B_shape,
                                                    reinterpret_cast<char*>(Y),
                                                    reinterpret_cast<char*>(Ho),
                                                    activation_f,
                                                    activation_g,
                                                    clip,
                                                    direction,
                                                    linear_before_reset);
}


template<> Converter::Conversion::Ptr Converter::Convert(const opset::GRUSequence& node) {
    auto make = [&] (auto refFunction) {
    return this->MakeConversion(refFunction,
                                node.input(0),
                                node.get_input_shape(0),
                                node.input(1),
                                node.get_input_shape(1),
                                node.input(2),
                                node.get_input_shape(2),
                                node.input(3),
                                node.get_input_shape(3),
                                node.input(4),
                                node.get_input_shape(4),
                                node.input(5),
                                node.get_input_shape(5),
                                node.output(0),
                                node.output(1),
                                node.get_activations()[0],
                                node.get_activations()[1],
                                node.get_clip(),
                                node.get_direction(),
                                node.get_linear_before_reset());
    };

    return CallSwitch(
        AP_WRAP(make, wrap_gru_sequence),
        node.input(0), allTypes,
        node.input(2), indexTypes);
}

template <typename T, typename U>
void wrap_rnn_sequence(const T* X,
                        const ngraph::Shape& X_shape,
                        const T* H,
                        const ngraph::Shape& H_shape,
                        const U* seq_lengths,
                        const ngraph::Shape& seq_lengths_shape,
                        const T* W,
                        const ngraph::Shape& W_shape,
                        const T* R,
                        const ngraph::Shape& R_shape,
                        const T* B,
                        const ngraph::Shape& B_shape,
                        T* Y,
                        T* Ho,
                        const std::string& activation_f,
                        float clip,
                        const ngraph::op::RecurrentSequenceDirection direction) {
     ngraph::runtime::reference::rnn_sequence<T, U>(reinterpret_cast<const char*>(X),
                                                    X_shape,
                                                    reinterpret_cast<const char*>(H),
                                                    H_shape,
                                                    reinterpret_cast<const char*>(seq_lengths),
                                                    seq_lengths_shape,
                                                    reinterpret_cast<const char*>(W),
                                                    W_shape,
                                                    reinterpret_cast<const char*>(R),
                                                    R_shape,
                                                    reinterpret_cast<const char*>(B),
                                                    B_shape,
                                                    reinterpret_cast<char*>(Y),
                                                    reinterpret_cast<char*>(Ho),
                                                    activation_f,
                                                    clip,
                                                    direction);
}


template<> Converter::Conversion::Ptr Converter::Convert(const opset::RNNSequence& node) {
    auto make = [&] (auto refFunction) {
    return this->MakeConversion(refFunction,
                                node.input(0),
                                node.get_input_shape(0),
                                node.input(1),
                                node.get_input_shape(1),
                                node.input(2),
                                node.get_input_shape(2),
                                node.input(3),
                                node.get_input_shape(3),
                                node.input(4),
                                node.get_input_shape(4),
                                node.input(5),
                                node.get_input_shape(5),
                                node.output(0),
                                node.output(1),
                                node.get_activations()[0],
                                node.get_clip(),
                                node.get_direction());
    };

    return CallSwitch(
        AP_WRAP(make, wrap_rnn_sequence),
        node.input(0), allTypes,
        node.input(2), indexTypes);
}

} // namespace ArmPlugin