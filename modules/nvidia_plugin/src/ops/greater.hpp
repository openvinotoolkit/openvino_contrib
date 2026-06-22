// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "comparison.hpp"

namespace ov {
namespace nvidia_gpu {

class GreaterOp : public Comparison {
public:
    GreaterOp(const CreationContext& context,
              const ov::Node& node,
              IndexCollection&& inputIds,
              IndexCollection&& outputIds);
};

}  // namespace nvidia_gpu
}  // namespace ov
