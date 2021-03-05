// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph_opset.hpp"
#include "utils.hpp"

namespace ArmPlugin {
namespace opset {

class ArmConcat : public Concat {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"ArmConcat", 0};
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info; }
    ArmConcat() = default;
    ~ArmConcat() override;

    ArmConcat(const ngraph::OutputVector& args, int64_t axis);

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
};
}  // namespace opset
}  // namespace ArmPlugin
