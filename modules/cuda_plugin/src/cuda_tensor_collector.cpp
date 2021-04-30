// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_tensor_collector.hpp"

namespace CUDAPlugin {

TensorCollector::TensorCollector(const std::vector<std::shared_ptr<ngraph::Node>>& nodes) {
}

TensorCollector::TensorIds
TensorCollector::getTensorIds(std::shared_ptr<ngraph::Node> node) const {
    return TensorIds{};
}

} // namespace CUDAPlugin
