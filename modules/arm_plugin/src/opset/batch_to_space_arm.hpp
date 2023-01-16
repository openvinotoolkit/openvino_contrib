// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph_opset.hpp"
#include "utils.hpp"

namespace ArmPlugin {
namespace opset {

namespace v1 {

class ArmBatchToSpace : public ov::op::v1::BatchToSpace {
public:
    OPENVINO_OP("ArmBatchToSpace", "arm_opset", ov::op::v1::BatchToSpace, 1);
    ArmBatchToSpace(const ov::Output<Node>& data,
                    const ov::Output<Node>& block_shape,
                    const ov::Output<Node>& crops_begin,
                    const ov::Output<Node>& crops_end);
};

}

}  // namespace opset
}  // namespace ArmPlugin
