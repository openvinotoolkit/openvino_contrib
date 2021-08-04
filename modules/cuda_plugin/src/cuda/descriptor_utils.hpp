// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/node.hpp>
#include <cuda/dnn.hpp>

namespace CUDA {

DnnTensorDescriptor makeDnnTensorDescr(const ngraph::element::Type& type,
        const ngraph::Shape& shape);

CUDA::DnnTensorDescriptor makeInputDnnTensorDescr(const ngraph::Node& node, int n);

CUDA::DnnTensorDescriptor makeOutputDnnTensorDescr(const ngraph::Node& node, int n);

} // namespace CUDA
