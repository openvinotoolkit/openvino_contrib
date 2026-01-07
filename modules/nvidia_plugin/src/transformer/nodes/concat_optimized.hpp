// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/concat.hpp>

namespace ov::nvidia_gpu::nodes {

class ConcatOptimized : public ov::op::v0::Concat {
public:
    using ov::op::v0::Concat::Concat;

    OPENVINO_OP("ConcatOptimized", "nvidia_gpu", ov::op::v0::Concat);

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        return std::make_shared<ConcatOptimized>(new_args, m_axis);
    }
};
}  // namespace ov::nvidia_gpu::nodes
