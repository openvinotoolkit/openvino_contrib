// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::nvidia_gpu::pass {

class NoopBroadcastTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("NoopBroadcastTransformation", "0");
    NoopBroadcastTransformation();
};

}  // namespace ov::nvidia_gpu::pass
