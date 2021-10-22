// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "comparison.hpp"

namespace CUDAPlugin {

class LessOp : public Comparison {
public:
    LessOp(const CreationContext& context,
           const ngraph::Node& node,
           IndexCollection&& inputIds,
           IndexCollection&& outputIds);
};

}  // namespace CUDAPlugin
