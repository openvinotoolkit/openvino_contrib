// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph_opset.hpp"
#include "utils.hpp"

namespace ArmPlugin {
namespace opset {

class ArmConvert : public Convert {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"ArmConvert", 0};
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info; }
    ArmConvert() = default;
    ~ArmConvert() override;

    ArmConvert(const ngraph::Output<ngraph::Node>& data, const ngraph::element::Type& destination_type);

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
};
}  // namespace opset
}  // namespace ArmPlugin
