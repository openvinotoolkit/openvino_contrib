// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda/dnn.hpp>
#include <ngraph/node.hpp>

namespace CUDA {

DnnTensorDescriptor makeDnnTensorDescr(const ov::element::Type& type, const ov::Shape& shape);

CUDA::DnnTensorDescriptor makeInputDnnTensorDescr(const ov::Node& node, int n);

CUDA::DnnTensorDescriptor makeOutputDnnTensorDescr(const ov::Node& node, int n);

}  // namespace CUDA
