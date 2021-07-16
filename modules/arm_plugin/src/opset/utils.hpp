// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include "arm_compute/core/QuantizationInfo.h"
#include "arm_compute/core/Types.h"

namespace ArmPlugin {
namespace opset {
float round(const float v);

arm_compute::QuantizationInfo makeQuantizationInfo(
                const ngraph::Output<ngraph::Node>& input_low,
                const ngraph::Output<ngraph::Node>& input_high,
                const ngraph::Output<ngraph::Node>& output_low,
                const ngraph::Output<ngraph::Node>& output_high);

arm_compute::ActivationLayerInfo makeActivationLayerInfo(ngraph::Node* node);


}  // namespace opset
}  // namespace ArmPlugin

namespace ngraph {
template <>
struct NGRAPH_API VariantWrapper<arm_compute::QuantizationInfo> : public VariantImpl<arm_compute::QuantizationInfo> {
    NGRAPH_RTTI_DECLARATION;
    VariantWrapper(const arm_compute::QuantizationInfo& value) : VariantImpl<arm_compute::QuantizationInfo>{value} {}
    ~VariantWrapper() override;
};

template <>
struct NGRAPH_API VariantWrapper<arm_compute::ActivationLayerInfo> : public VariantImpl<arm_compute::ActivationLayerInfo> {
    NGRAPH_RTTI_DECLARATION;
    VariantWrapper(const arm_compute::ActivationLayerInfo& value) : VariantImpl<arm_compute::ActivationLayerInfo>{value} {}
    ~VariantWrapper() override;
};
}  // namespace ngraph
