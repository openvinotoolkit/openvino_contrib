// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "depth_to_space_arm.hpp"
#include <sstream>

using namespace std;
using namespace ov;

ArmPlugin::opset::v0::ArmDepthToSpace::ArmDepthToSpace(const ov::Output<Node>& data, const DepthToSpaceMode& mode,
                                                       std::size_t block_size, ov::PartialShape output_shape)
        : m_output_shape{std::move(output_shape)} {
    set_arguments({data});
    ov::op::v0::DepthToSpace::m_mode = mode;
    ov::op::v0::DepthToSpace::m_blocksize = block_size;
    constructor_validate_and_infer_types();
}

void ArmPlugin::opset::v0::ArmDepthToSpace::validate_and_infer_types() {
    if (m_output_shape == PartialShape{}) {
        ov::op::v0::DepthToSpace::validate_and_infer_types();
    } else {
        set_output_type(0, get_input_element_type(0), m_output_shape);
    }
}

std::shared_ptr<ov::Node>
ArmPlugin::opset::v0::ArmDepthToSpace::clone_with_new_inputs(const OutputVector &new_args) const {
    return std::make_shared<ArmDepthToSpace>(new_args.at(0), get_mode(), get_block_size(), m_output_shape);
}
