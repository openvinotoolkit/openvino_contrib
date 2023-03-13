// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "complex_mul.hpp"
#include <openvino/core/parallel.hpp>
#include <ie_common.h>

using namespace TemplateExtension;

ComplexMultiplication::ComplexMultiplication(const ov::OutputVector& args) : Op(args) {
    constructor_validate_and_infer_types();
}

void ComplexMultiplication::validate_and_infer_types() {
    auto outShape = get_input_partial_shape(0);
    set_output_type(0, get_input_element_type(1), outShape);
}

std::shared_ptr<ov::Node> ComplexMultiplication::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 2, "Incorrect number of new arguments");
    return std::make_shared<ComplexMultiplication>(new_args);
}

bool ComplexMultiplication::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const float* inp0 = reinterpret_cast<float*>(inputs[0].data());
    const float* inp1 = reinterpret_cast<float*>(inputs[1].data());
    float* out = reinterpret_cast<float*>(outputs[0].data());

    size_t channels0 = inputs[0].get_shape()[1];
    size_t channels1 = inputs[1].get_shape()[1];
    size_t batch = inputs[0].get_shape()[0];
    size_t spatialSize = inputs[0].get_shape()[2] * inputs[0].get_shape()[3];

    // x1 = x_r * y_r - x_i * y_i
    // x2 = x_r * y_i + x_i * y_r
    if (channels0 == channels1)
        ov::parallel_for(channels0 * batch, [&](size_t ch) {
            for (size_t i = 0; i < spatialSize; ++i) {
                    size_t outIdx = (ch * spatialSize + i) * 2;
                    float real0 = inp0[outIdx];
                    float imag0 = inp0[outIdx + 1];
                    float real1 = inp1[outIdx];
                    float imag1 = inp1[outIdx + 1];
                    out[outIdx] = real0 * real1 - imag0 * imag1;
                    out[outIdx + 1] = real0 * imag1 + imag0 * real1;
            }
        });
    else if (channels1 == 1)
        ov::parallel_for(channels0 * batch, [&](size_t ch) {
            size_t b = ch / channels0;
            for (size_t i = 0; i < spatialSize; ++i) {
                size_t outIdx = (ch * spatialSize + i) * 2;
                size_t inpIdx = (b * spatialSize + i) * 2;
                float real0 = inp0[outIdx];
                float imag0 = inp0[outIdx + 1];
                float real1 = inp1[inpIdx];
                float imag1 = inp1[inpIdx + 1];
                out[outIdx] = real0 * real1 - imag0 * imag1;
                out[outIdx + 1] = real0 * imag1 + imag0 * real1;
            }
        });
    else
        IE_THROW() << "Wrong number of channels for second input!";

    return true;
}

bool ComplexMultiplication::has_evaluate() const {
    for (size_t i = 0; i < get_input_size(); ++i)
        if (get_input_element_type(i) != ngraph::element::f32)
            return false;
    return true;
}
