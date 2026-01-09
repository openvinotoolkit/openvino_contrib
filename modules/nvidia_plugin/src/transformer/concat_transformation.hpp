// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::nvidia_gpu::pass {

class ConcatTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConcatTransformation", "0");
    ConcatTransformation();
};

}  // namespace ov::nvidia_gpu::pass
