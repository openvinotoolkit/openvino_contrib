// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph_opset.hpp"
#include "utils.hpp"

namespace ArmPlugin {
namespace opset {
namespace v0 {
class ArmDepthToSpace : public ov::op::v0::DepthToSpace {
public:
    OPENVINO_OP("ArmDepthToSpace", "arm_opset", ov::op::v0::DepthToSpace, 0);
    ArmDepthToSpace(const ov::Output<Node>& data, const DepthToSpaceMode& mode,
                    std::size_t block_size = 1, ov::PartialShape output_shape = {});

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    void validate_and_infer_types() override;

private:
    ov::PartialShape m_output_shape;
};
}  // namespace v0
}  // namespace opset
}  // namespace ArmPlugin

