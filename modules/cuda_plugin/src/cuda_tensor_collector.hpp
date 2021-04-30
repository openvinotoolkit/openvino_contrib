// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/node.hpp>

namespace CUDAPlugin {

class TensorCollector {
 public:
  struct TensorIds {
    std::vector<unsigned> inputs_;
    std::vector<unsigned> outputs_;
  };

  explicit TensorCollector(const std::vector<std::shared_ptr<ngraph::Node>>& nodes);
  [[nodiscard]]
  TensorIds getTensorIds(std::shared_ptr<ngraph::Node> node) const;
};

} // namespace CUDAPlugin
