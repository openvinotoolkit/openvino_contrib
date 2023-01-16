// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph_opset.hpp"
#include "utils.hpp"

namespace ArmPlugin {
namespace opset {
namespace v5 {
class ArmBatchNormInference : public ov::op::v5::BatchNormInference {
public:
    OPENVINO_OP("ArmBatchNormInference", "arm_opset", ov::op::v5::BatchNormInference, 5);
    ArmBatchNormInference(const ov::Output<Node>& input,
                          const ov::Output<Node>& gamma,
                          const ov::Output<Node>& beta,
                          const ov::Output<Node>& mean,
                          const ov::Output<Node>& variance,
                          double epsilon,
                          ov::PartialShape output_shape = {});

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    void validate_and_infer_types() override;

private:
    ov::PartialShape m_output_shape;
};
}  // namespace v5
}  // namespace opset
}  // namespace ArmPlugin

