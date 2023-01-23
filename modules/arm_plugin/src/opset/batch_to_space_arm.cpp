// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batch_to_space_arm.hpp"

#include <utility>
#include "openvino/op/util/precision_sensitive_attribute.hpp"

ArmPlugin::opset::v1::ArmBatchToSpace::ArmBatchToSpace(const ov::Output<Node> &data,
                                                       const ov::Output<Node> &block_shape,
                                                       const ov::Output<Node> &crops_begin,
                                                       const ov::Output<Node> &crops_end,
                                                       ov::PartialShape  output_shape) : m_output_shape(std::move(output_shape)) {
    set_arguments({data, block_shape, crops_begin, crops_end});
    ov::mark_as_precision_sensitive(input(1));
    ov::mark_as_precision_sensitive(input(2));
    ov::mark_as_precision_sensitive(input(3));
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node>
ArmPlugin::opset::v1::ArmBatchToSpace::clone_with_new_inputs(const ov::OutputVector &new_args) const {
    return std::make_shared<ArmBatchToSpace>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}

void ArmPlugin::opset::v1::ArmBatchToSpace::validate_and_infer_types() {
    if (m_output_shape == ov::PartialShape{}) {
        ov::op::v1::BatchToSpace::validate_and_infer_types();
    } else {
        set_output_type(0, get_input_element_type(0), m_output_shape);
    }
}

