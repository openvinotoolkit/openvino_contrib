// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::nvidia_gpu::pass {

class RemoveDuplicatedResultsTransformation : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("RemoveDuplicatedResultsTransformation", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& f) override;
};

}  // namespace ov::nvidia_gpu::pass
