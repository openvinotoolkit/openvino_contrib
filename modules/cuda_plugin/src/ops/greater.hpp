// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "comparison.hpp"

namespace CUDAPlugin {

class GreaterOp : public Comparison {
public:
    GreaterOp(const CreationContext& context,
              const ov::Node& node,
              IndexCollection&& inputIds,
              IndexCollection&& outputIds);
};

}  // namespace CUDAPlugin
