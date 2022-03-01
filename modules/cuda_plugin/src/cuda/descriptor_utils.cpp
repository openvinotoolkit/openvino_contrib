// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "descriptor_utils.hpp"

#include <array>
#include <ops/converters.hpp>

namespace CUDA {

DnnTensorDescriptor makeDnnTensorDescr(const ngraph::element::Type& type, const ngraph::Shape& shape) {
    Expects(!shape.empty());
    Expects(shape.size() <= CUDNN_DIM_MAX);
    std::vector<int> dims;
    std::transform(shape.begin(), shape.end(), std::back_inserter(dims), [](auto v) { return static_cast<int>(v); });
    const int CUDNN_DIM_MIN =
        4;  // see note here: https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetTensorNdDescriptor
    while (dims.size() < CUDNN_DIM_MIN) {
        dims.push_back(1);
    }
    decltype(dims) strides(dims.size(), 0);
    strides.back() = 1;
    for (int i = dims.size() - 1; i > 0; i--) strides[i - 1] = strides[i] * dims[i];
    return DnnTensorDescriptor{}.set(
        CUDAPlugin::convertDataType<cudnnDataType_t>(type), dims.size(), dims.data(), strides.data());
}

CUDA::DnnTensorDescriptor makeInputDnnTensorDescr(const ngraph::Node& node, int n) {
    return makeDnnTensorDescr(node.get_input_element_type(n), node.get_input_shape(n));
}

CUDA::DnnTensorDescriptor makeOutputDnnTensorDescr(const ngraph::Node& node, int n) {
    return makeDnnTensorDescr(node.get_output_element_type(n), node.get_output_shape(n));
}

}  // namespace CUDA
