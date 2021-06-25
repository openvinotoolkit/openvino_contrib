// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph_opset.hpp"
#include "utils.hpp"

namespace ArmPlugin {
namespace opset {

class ArmSplit : public Split {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"ArmSplit", 0};
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info; }
    ArmSplit() = default;
    ~ArmSplit() override;

    ArmSplit(const ngraph::Output<ngraph::Node>& data, const ngraph::Output<ngraph::Node>& axis, const size_t num_splits);

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
};
}  // namespace opset
}  // namespace ArmPlugin
