// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/concat.hpp>

namespace CUDAPlugin::nodes {

class ConcatOptimized : public ov::op::v0::Concat {
   public:
    using ov::op::v0::Concat::Concat;

    inline static constexpr type_info_t type_info{"ConcatOptimized", 0ul};
    const type_info_t& get_type_info() const override { return type_info; }

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        return std::make_shared<ConcatOptimized>(new_args, m_axis);
    }
};
}  // namespace CUDAPlugin::nodes
