// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grid_sample.hpp"
#include <openvino/core/parallel.hpp>

using namespace TemplateExtension;

GridSample::GridSample(const ov::OutputVector& args) : Op(args) {
    constructor_validate_and_infer_types();
}

void GridSample::validate_and_infer_types() {
    auto outShape = get_input_partial_shape(0);  // NC
    // Grid input has a shape NxHxWx2
    auto gridShape = get_input_partial_shape(1);
    outShape[2] = gridShape[1];  // H
    outShape[3] = gridShape[2];  // W
    set_output_type(0, get_input_element_type(0), outShape);
}

std::shared_ptr<ov::Node> GridSample::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 2, "Incorrect number of new arguments");
    return std::make_shared<GridSample>(new_args);
}

bool GridSample::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const float *inpData = reinterpret_cast<const float *>(inputs[0].data());
    const float *gridData = reinterpret_cast<const float *>(inputs[1].data());
    float* outData = reinterpret_cast<float*>(outputs[0].data());

    std::vector<size_t> inpDims = inputs[0].get_shape();
    std::vector<size_t> outDims = outputs[0].get_shape();

    const size_t batch     = outDims[0];
    const size_t channels  = outDims[1];
    const size_t height    = outDims[2];
    const size_t width     = outDims[3];
    const size_t inpHeight = inpDims[2];
    const size_t inpWidth  = inpDims[3];
    const size_t inpPlane  = inpHeight * inpWidth;
    const size_t outPlane  = height * width;

    std::vector<float> zerosPlane(inpDims[1] * inpDims[2] * inpDims[3], 0);
    float* zeros = zerosPlane.data();

    ov::parallel_for(batch, [&](size_t d) {
        const float* inp  = inpData + d * channels * inpPlane;
        const float* grid = gridData + d * outPlane * 2;
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                size_t offset = y * width + x;

                float input_x = 0.5f * (grid[offset * 2] + 1) * (inpWidth - 1);
                int x0 = static_cast<int>(std::floor(input_x));
                int x1 = x0 + 1;

                float input_y = 0.5f * (grid[offset * 2 + 1] + 1) * (inpHeight - 1);
                int y0 = static_cast<int>(std::floor(input_y));
                int y1 = y0 + 1;

                const float* inp_row0 = (0 <= y0 && y0 < inpHeight) ? inp + y0 * inpWidth : zeros;
                const float* inp_row1 = (0 <= y1 && y1 < inpHeight) ? inp + y1 * inpWidth : zeros;
                float* out = outData + d * channels * outPlane;
                if ((x1 < 0 || inpWidth <= x1) && (x0 < 0 || inpWidth <= x0)) {
                    for (size_t c = 0; c < channels; ++c) {
                        out[offset] = 0;
                        out += outPlane;
                    }
                }
                else if (x1 < 0 || inpWidth <= x1) {
                    for (size_t c = 0; c < channels; ++c) {
                        out[offset] = inp_row0[x0] +
                            (input_y - y0) * (inp_row1[x0] - inp_row0[x0]) +
                            (input_x - x0) * (-inp_row0[x0] +
                            (input_y - y0) * (inp_row0[x0] - inp_row1[x0]));
                        out += outPlane;
                        inp_row0 += inpPlane;
                        inp_row1 += inpPlane;
                    }
                }
                else if (x0 < 0 || inpWidth <= x0) {
                    for (size_t c = 0; c < channels; ++c) {
                        out[offset] =
                            (input_x - x0) * (inp_row0[x1] + (input_y - y0) * (inp_row1[x1] - inp_row0[x1]));
                        out += outPlane;
                        inp_row0 += inpPlane;
                        inp_row1 += inpPlane;
                    }
                } else {
                    for (size_t c = 0; c < channels; ++c) {
                        out[offset] = inp_row0[x0] +
                            (input_y - y0) * (inp_row1[x0] - inp_row0[x0]) +
                            (input_x - x0) * (inp_row0[x1] - inp_row0[x0] +
                            (input_y - y0) * (inp_row1[x1] - inp_row0[x1] - inp_row1[x0] + inp_row0[x0]));
                        out += outPlane;
                        inp_row0 += inpPlane;
                        inp_row1 += inpPlane;
                    }
                }
            }
        }
    });
    return true;
}

bool GridSample::has_evaluate() const {
    for (size_t i = 0; i < get_input_size(); ++i)
        if (get_input_element_type(i) != ov::element::f32)
            return false;
    return true;
}
