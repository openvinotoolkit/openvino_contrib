// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda/dnn.hpp>

#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace CUDA {

DnnTensorDescriptor makeDnnTensorDescr(const ov::element::Type& type, const ov::Shape& shape);

CUDA::DnnTensorDescriptor makeInputDnnTensorDescr(const ov::Node& node, int n);

CUDA::DnnTensorDescriptor makeOutputDnnTensorDescr(const ov::Node& node, int n);

}  // namespace CUDA
