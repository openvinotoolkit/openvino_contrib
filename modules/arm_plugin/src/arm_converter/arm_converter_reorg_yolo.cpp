// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <details/ie_exception.hpp>

#include <arm_compute/runtime/NEON/functions/NEReorgLayer.h>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template <> Converter::Conversion::Ptr Converter::Convert(const opset::ReorgYolo& node) {
    auto strides = node.get_strides();
    if (strides[0] != strides[1] && strides[0] != 1) {
        THROW_IE_EXCEPTION << "Reorg op doesn't support asymmetric strides";
    }
    int32_t stride = static_cast<int32_t>(strides[0]);
    return MakeConversion<arm_compute::NEReorgLayer>(node.input(0), node.output(0), stride);
}
}  //  namespace ArmPlugin
