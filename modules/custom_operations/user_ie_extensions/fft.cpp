// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fft.hpp"

#include <openvino/core/parallel.hpp>
#include <opencv2/core/core_c.h>

using namespace TemplateExtension;

void fftshift(CvMat* src, bool inverse) {
    // tl | tr        br | bl
    // ---+---   ->   ---+---
    // bl | br        tr | tl

    float* data;
    int step;
    CvSize size;
    cvGetRawData(src, (uchar**)&data, &step, &size);

    int height = size.height;
    int width = size.width;
    int h2 = height / 2;
    int w2 = width / 2;

    if (height % 2 || width % 2) {
        // Swap rows.
        CvMat* srcTop = new CvMat();
        CvMat* srcBot = new CvMat();
        CvMat* dstTop = new CvMat();
        CvMat* dstBot = new CvMat();
        int topH = inverse ? h2 : (h2 + height % 2);
        int botH = height - topH;
        cvInitMatHeader(srcTop, topH, width, CV_32FC2, data, step);
        cvInitMatHeader(srcBot, botH, width, CV_32FC2, data + topH * width * 2, step);
        cvInitMatHeader(dstTop, topH, width, CV_32FC2, data + botH * width * 2, step);
        cvInitMatHeader(dstBot, botH, width, CV_32FC2, data, step);

        CvMat* tmp = cvCloneMat(srcTop);
        cvCopy(srcBot, dstBot, 0);
        cvCopy(tmp, dstTop, 0);

        cvReleaseMat(&tmp);
        delete srcTop;
        delete srcBot;
        delete dstTop;
        delete dstBot;

        // Swap columns.
        CvMat* srcL = new CvMat();
        CvMat* srcR = new CvMat();
        CvMat* dstL = new CvMat();
        CvMat* dstR = new CvMat();
        int leftW = inverse ? w2 : (w2 + width % 2);
        int rightW = width - leftW;

        cvInitMatHeader(srcL, height, leftW, CV_32FC2, data, step);
        cvInitMatHeader(srcR, height, rightW, CV_32FC2, data + leftW * 2, step);
        cvInitMatHeader(dstL, height, leftW, CV_32FC2, data + rightW * 2, step);
        cvInitMatHeader(dstR, height, rightW, CV_32FC2, data, step);

        tmp = cvCloneMat(srcL);
        cvCopy(srcR, dstR, 0);
        cvCopy(tmp, dstL, 0);

        cvReleaseMat(&tmp);
        delete srcL;
        delete srcR;
        delete dstL;
        delete dstR;

        return;
    }

    CvMat* tl = new CvMat();
    CvMat* tr = new CvMat();
    CvMat* bl = new CvMat();
    CvMat* br = new CvMat();

    cvInitMatHeader(tl, h2, w2, CV_32FC2, data, step);
    cvInitMatHeader(tr, h2, w2, CV_32FC2, data + width, step);
    cvInitMatHeader(bl, h2, w2, CV_32FC2, data + height * width, step);
    cvInitMatHeader(br, h2, w2, CV_32FC2, data + height * width + width, step);

    CvArr* mask = 0;
    CvMat* tmp = cvCloneMat(tl);
    cvCopy(br, tl, mask);
    cvCopy(tmp, br, mask);

    cvCopy(tr, tmp, mask);
    cvCopy(bl, tr, mask);
    cvCopy(tmp, bl, mask);

    cvReleaseMat(&tmp);

    delete tl;
    delete tr;
    delete bl;
    delete br;
}

FFT::FFT(const ov::OutputVector& args, bool inverse, bool centered) : Op(args) {
    constructor_validate_and_infer_types();
    this->inverse = inverse;
    this->centered = centered;
}

void FFT::validate_and_infer_types() {
    auto outShape = get_input_partial_shape(0);
    set_output_type(0, get_input_element_type(0), outShape);
}

std::shared_ptr<ov::Node> FFT::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 2, "Incorrect number of new arguments, provided: ", new_args.size());
    return std::make_shared<FFT>(new_args, inverse, centered);
}

bool FFT::visit_attributes(ov::AttributeVisitor& visitor) {
    int inverse_i = static_cast<int>(inverse);
    int centered_i = static_cast<int>(centered);
    visitor.on_attribute("inverse", inverse_i);
    visitor.on_attribute("centered", centered_i);
    inverse = static_cast<bool>(inverse_i);
    centered = static_cast<bool>(centered_i);
    return true;
}

bool FFT::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    //const_cast because the cvSetData use user pointer as non-const, should be ok as it looks like input data
    auto *inpData = const_cast<float*>(inputs[0].data<float>());

    if (inputs[1].get_element_type() != ov::element::i32)
        OPENVINO_THROW("Unexpected dims type: " + inputs[1].get_element_type().to_string());

    auto *signalDimsData = inputs[1].data<int32_t>();
    auto *outData = outputs[0].data<float>();
    std::vector<size_t> dims = inputs[0].get_shape();
    const size_t numSignalDims = inputs[1].get_shape().empty() ? 1:  inputs[1].get_shape().size();

    if (!((dims.size() == 3 && numSignalDims == 1 && signalDimsData[0] == 1) ||
          (dims.size() == 4 && ((numSignalDims == 1 && signalDimsData[0] == 1) ||
                                (numSignalDims == 2 && signalDimsData[0] == 1 && signalDimsData[1] == 2))) ||
          (dims.size() == 5 && ((numSignalDims == 2 && signalDimsData[0] == 1 && signalDimsData[1] == 2) ||
                                (numSignalDims == 2 && signalDimsData[0] == 2 && signalDimsData[1] == 3))))) {
        std::ostringstream ss;
        for (size_t i = 0; i < numSignalDims; ++i)
            ss << signalDimsData[i] << " ";
        OPENVINO_THROW("Unsupported configuration: Input dims " + std::to_string(dims.size()) + " and signal dims " + ss.str());
    }

    const int batch = dims[0];

    if (dims.size() == 5 && numSignalDims == 2 && signalDimsData[0] == 1 && signalDimsData[1] == 2) {
        const int channels = dims[1];
        int rows = dims[2];
        int cols = dims[3];
        const int planeSize = channels * rows * cols;
        ov::parallel_for(batch * cols, [&](size_t d) {
            int b = d / cols;
            int col = d % cols;
            // Copy a slice from input
            CvMat* inpSlice = cvCreateMatHeader(channels * rows, 1, CV_32FC2);
            CvMat* outSlice = cvCreateMatHeader(channels * rows, 1, CV_32FC2);
            cvSetData(inpSlice, reinterpret_cast<void*>(inpData + (b * planeSize + col) * 2), cols * 2 * sizeof(float));
            cvSetData(outSlice, reinterpret_cast<void*>(outData + (b * planeSize + col) * 2), cols * 2 * sizeof(float));

            CvMat* inp_col = cvCloneMat(inpSlice);

            CvMat inp_header, *inp;
            inp = cvReshape(inp_col, &inp_header, 2, channels);

            CvMat* out = cvCreateMatHeader(channels, rows, CV_32FC2);
            cvCreateData(out);

            if (centered)
                fftshift(inp, true);

            if (inverse)
                cvDFT(inp, out, CV_DXT_INVERSE, 0);
            else
                cvDFT(inp, out, CV_DXT_FORWARD, 0);
            cvScale(out, out, 1.0 / sqrtf(channels * rows), 0);

            if (centered)
                fftshift(out, false);

            CvMat out_col_header, *out_col;
            out_col = cvReshape(out, &out_col_header, 2, channels * rows);

            cvCopy(out_col, outSlice, 0);

            cvReleaseData(inp_col);
            cvReleaseMat(&inp_col);

            cvReleaseData(out);
            cvReleaseMat(&out);

            cvReleaseMat(&inpSlice);
            cvReleaseMat(&outSlice);
        });
    } else if (dims.size() == 5 && numSignalDims == 2 && signalDimsData[0] == 2 && signalDimsData[1] == 3) {
        const int channels = dims[1];
        int rows = dims[2];
        int cols = dims[3];
        int planeSize = rows * cols * 2;  // 2 is last dimension size
        ov::parallel_for(batch * channels, [&](size_t d) {
            CvMat* inp = cvCreateMatHeader(rows, cols, CV_32FC2);
            CvMat* out = cvCreateMatHeader(rows, cols, CV_32FC2);
            cvSetData(inp, reinterpret_cast<void*>(inpData + d * planeSize), cols * 2 * sizeof(float));
            cvSetData(out, reinterpret_cast<void*>(outData + d * planeSize), cols * 2 * sizeof(float));

            if (centered)
                fftshift(inp, true);

            if (inverse)
                cvDFT(inp, out, CV_DXT_INVERSE, 0);
            else
                cvDFT(inp, out, CV_DXT_FORWARD, 0);
            cvScale(out, out, 1.0 / sqrtf(cols * rows), 0);

            if (centered)
                fftshift(out, false);

            cvReleaseMat(&inp);
            cvReleaseMat(&out);
        });
    } else if (dims.size() == 4 && numSignalDims == 2 && signalDimsData[0] == 1 && signalDimsData[1] == 2) {
        int rows = dims[1];
        int cols = dims[2];
        int planeSize = rows * cols * 2;  // 2 is last dimension size
        ov::parallel_for(batch, [&](size_t d) {
            CvMat* inp = cvCreateMatHeader(rows, cols, CV_32FC2);
            CvMat* out = cvCreateMatHeader(rows, cols, CV_32FC2);
            cvSetData(inp, reinterpret_cast<void*>(inpData + d * planeSize), cols * 2 * sizeof(float));
            cvSetData(out, reinterpret_cast<void*>(outData + d * planeSize), cols * 2 * sizeof(float));

            if (centered)
                fftshift(inp, true);

            if (inverse)
                cvDFT(inp, out, CV_DXT_INVERSE, 0);
            else
                cvDFT(inp, out, CV_DXT_FORWARD, 0);
            cvScale(out, out, 1.0 / sqrtf(cols * rows), 0);

            if (centered)
                fftshift(out, false);

            cvReleaseMat(&inp);
            cvReleaseMat(&out);
        });
    } else if (dims.size() == 4 && numSignalDims == 1 && signalDimsData[0] == 1) {
        int rows = dims[1];
        int cols = dims[2];

        const int planeSize = rows;
        ov::parallel_for(batch * cols, [&](size_t d) {
            int b = d / cols;
            int col = d % cols;
            CvMat* inp = cvCreateMatHeader(rows, 1, CV_32FC2);
            CvMat* out = cvCreateMatHeader(rows, 1, CV_32FC2);
            cvSetData(inp, reinterpret_cast<void*>(inpData + (b * planeSize * cols + col) * 2), cols * 2 * sizeof(float));
            cvSetData(out, reinterpret_cast<void*>(outData + (b * planeSize * cols + col) * 2), cols * 2 * sizeof(float));

            if (centered)
                fftshift(inp, true);

            if (inverse)
                cvDFT(inp, out, CV_DXT_INVERSE, 0);
            else
                cvDFT(inp, out, CV_DXT_FORWARD, 0);
            cvScale(out, out, 1.0 / sqrtf(rows), 0);

            if (centered)
                fftshift(out, false);

            cvReleaseMat(&inp);
            cvReleaseMat(&out);
        });
    } else if (dims.size() == 3) {
        int rows = dims[0];
        int cols = dims[1];
        CvMat* inp = cvCreateMatHeader(rows, cols, CV_32FC2);
        CvMat* out = cvCreateMatHeader(rows, cols, CV_32FC2);
        cvSetData(inp, reinterpret_cast<void*>(inpData), cols * 2 * sizeof(float));
        cvSetData(out, reinterpret_cast<void*>(outData), cols * 2 * sizeof(float));

        if (inverse)
            cvDFT(inp, out, CV_DXT_INVERSE | CV_DXT_ROWS, 0);
        else
            cvDFT(inp, out, CV_DXT_FORWARD | CV_DXT_ROWS, 0);
        cvScale(out, out, 1.0 / sqrtf(cols), 0);

        cvReleaseMat(&inp);
        cvReleaseMat(&out);
    }
    return true;
}

bool FFT::has_evaluate() const {
    if (get_input_element_type(0) == ov::element::f32 && get_input_element_type(1) == ov::element::i32)
        return true;
    return false;
}
