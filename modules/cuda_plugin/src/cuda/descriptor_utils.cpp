// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "descriptor_utils.hpp"

#include <array>
#include <ops/converters.hpp>

namespace CUDA {

std::array<int, 5> toArray(const ngraph::Shape& shape) {
    std::array<int, 5> a{1, 1, 1, 1, 1};
    if (shape.empty()) return a;
    for (std::size_t i = std::min(shape.size(), a.size()); i > 0;) {
        i--;
        a[i] = shape[i];
    }
    return a;
}

DnnTensorDescriptor makeDnnTensorDescr(
    const ngraph::element::Type& type,
    const ngraph::Shape& shape) {  // TODO: different ops have different shape limitations
    auto dims = toArray(shape);
    decltype(dims) strides;
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
