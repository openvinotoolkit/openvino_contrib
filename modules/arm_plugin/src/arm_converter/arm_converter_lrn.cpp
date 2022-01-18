// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <arm_compute/runtime/NEON/functions/NENormalizationLayer.h>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::LRN& node) {
    auto&& axes = node.get_reduction_axes().to_vector();
    uint32_t norm_size = node.get_nsize();
    float alpha        = node.get_alpha();
    float beta         = node.get_beta();
    float kappa        = node.get_bias();

    arm_compute::NormType norm_type;
    if (axes.size() == 1 && axes[0] == 1) {
       norm_type = arm_compute::NormType::CROSS_MAP;
    } else {
        for (size_t i = 0; i < axes.size(); i++) {
            if (axes[i] != i + 2) {
                IE_THROW() << "Unsupported mode of LRN layer";
            }
        }
       norm_type = arm_compute::NormType::IN_MAP_2D;
    }
    arm_compute::NormalizationLayerInfo info(norm_type, norm_size, alpha, beta, kappa);

    return MakeConversion<arm_compute::NENormalizationLayer>(node.input(0), node.output(0), info);
}
} //  namespace ArmPlugin
