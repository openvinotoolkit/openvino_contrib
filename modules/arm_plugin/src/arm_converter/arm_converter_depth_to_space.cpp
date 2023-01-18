// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <arm_compute/runtime/NEON/functions/NEDepthToSpaceLayer.h>
#include <ngraph/runtime/reference/depth_to_space.hpp>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::v0::ArmDepthToSpace& node) {
    if (node.get_input_shape(0).size() > 4 ||
        node.get_mode() != opset::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST) {
        auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.get_input_shape(0),
                                    node.output(0),
                                    node.get_output_shape(0),
                                    node.get_block_size(),
                                    node.get_mode(),
                                    node.get_element_type().size());
        };
        return make (ngraph::runtime::reference::depth_to_space);
    }

    int32_t block_shape = node.get_block_size();
    return MakeConversion<arm_compute::NEDepthToSpaceLayer>(node.input(0), node.output(0), block_shape);
}
} //  namespace ArmPlugin
