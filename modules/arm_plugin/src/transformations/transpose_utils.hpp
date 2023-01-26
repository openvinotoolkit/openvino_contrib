// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "opset/opset.hpp"

namespace ArmPlugin {
namespace pass {

std::shared_ptr<ArmPlugin::opset::ArmTranspose> transpose_on_input(const ov::Output<ov::Node>& input, size_t rank);
std::shared_ptr<ArmPlugin::opset::ArmTranspose> transpose_on_output(const ov::Output<ov::Node>& input, size_t rank);
ov::PartialShape transpose_output_shape(const std::shared_ptr<ov::Node>& node, size_t rank);
 
} // pass
} // ArmPlugin
