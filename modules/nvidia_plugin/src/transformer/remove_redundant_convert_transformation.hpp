// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::nvidia_gpu::pass {

class RemoveRedundantConvertTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("RemoveRedundantConvertTransformation", "0");
    RemoveRedundantConvertTransformation();
};

}  // namespace ov::nvidia_gpu::pass
