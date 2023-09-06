// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::nvidia_gpu::pass {

class RemoveRedundantConvertTransformation : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("RemoveRedundantConvertTransformation", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

class MergeSubsequentConvertTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MergeSubsequentConvertTransformation", "0");
    MergeSubsequentConvertTransformation();
};

}  // namespace ov::nvidia_gpu::pass
