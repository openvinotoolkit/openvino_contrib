// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <arm_compute/runtime/NEON/functions/NEDepthToSpaceLayer.h>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::DepthToSpace& node) {
    if (node.get_input_shape(0).size() > 4) {
        IE_THROW() << "Unsupported DepthToSpace with num dimensions > 4";
    }
    if (node.get_mode() != opset::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST) {
        IE_THROW() << "Unsupported DepthToSpace mode";
    }

    int32_t block_shape = node.get_block_size();
    return MakeConversion<arm_compute::NEDepthToSpaceLayer>(node.input(0), node.output(0), block_shape);
}
} //  namespace ArmPlugin
