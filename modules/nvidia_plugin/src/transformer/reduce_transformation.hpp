// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::nvidia_gpu::pass {

class ReduceTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReduceTransformation", "0");
    ReduceTransformation();
};

}  // namespace ov::nvidia_gpu::pass
