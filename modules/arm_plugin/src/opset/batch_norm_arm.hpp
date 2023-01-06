// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph_opset.hpp"
#include "utils.hpp"

namespace ArmPlugin {
namespace opset {
struct ChannelShapedInputSpec {
    ov::element::Type m_element_type;
    ov::PartialShape m_shape;
    std::string m_input_name;
};
static std::tuple<ov::element::Type, ov::PartialShape, ov::PartialShape> infer_batch_norm_forward(
        const ov::Node* node,
        ov::element::Type input_element_type,
        ov::element::Type gamma_element_type,
        ov::element::Type beta_element_type,
        ov::element::Type mean_element_type,
        ov::element::Type variance_element_type,
        const ov::PartialShape& input_shape,
        const ov::PartialShape& gamma_shape,
        const ov::PartialShape& beta_shape,
        const ov::PartialShape& mean_shape,
        const ov::PartialShape& variance_shape,
        DataLayout layout);
static std::tuple<ov::element::Type, ov::PartialShape, ov::PartialShape> infer_batch_norm_forward_helper(
        const ov::Node* node,
        ov::element::Type input_element_type,
        const ov::PartialShape& input_shape,
        const std::vector<ChannelShapedInputSpec>& channel_shaped_inputs,
        DataLayout layout);
class ArmBatchNormInference : public ov::op::Op {
public:
    OPENVINO_OP("ArmBatchNormInference", "arm_opset");
    ArmBatchNormInference() = default;
    ArmBatchNormInference(const ov::Output<Node>& input,
                          const ov::Output<Node>& gamma,
                          const ov::Output<Node>& beta,
                          const ov::Output<Node>& mean,
                          const ov::Output<Node>& variance,
                          double epsilon,
                          DataLayout layout = DataLayout::NCHW);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    double get_eps_value() const {
        return m_epsilon;
    }
    void set_eps_value(double epsilon) {
        m_epsilon = epsilon;
    }
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

private:
    static constexpr size_t INPUT_DATA = 0;
    static constexpr size_t INPUT_GAMMA = 1;
    static constexpr size_t INPUT_BETA = 2;
    static constexpr size_t INPUT_MEAN = 3;
    static constexpr size_t INPUT_VARIANCE = 4;

    double m_epsilon{0};
    DataLayout m_layout;
};
}  // namespace opset
}  // namespace ArmPlugin

