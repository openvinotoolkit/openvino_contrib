// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::nvidia_gpu::pass {

class DetectionOutputFixInputTypesTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("DetectionOutputFixInputTypesTransformation", "0");
    DetectionOutputFixInputTypesTransformation();
};

}  // namespace ov::nvidia_gpu::pass
