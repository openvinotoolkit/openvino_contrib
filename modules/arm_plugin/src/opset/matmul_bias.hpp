// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include <algorithm>
#include "quantize.hpp"

#include "ngraph_opset.hpp"

namespace ArmPlugin {
namespace opset {

class MatMulBias : public MatMul {
public:
    NGRAPH_RTTI_DECLARATION;
    MatMulBias() = default;
    ~MatMulBias() override;

    MatMulBias(const ngraph::Output<ngraph::Node>& data,
               const ngraph::Output<ngraph::Node>& weights,
               const ngraph::Output<ngraph::Node>& bias,
               const bool& transpose_b = false);

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;

private:
    bool m_transpose_b;
};

}  // namespace opset
}  // namespace ArmPlugin
