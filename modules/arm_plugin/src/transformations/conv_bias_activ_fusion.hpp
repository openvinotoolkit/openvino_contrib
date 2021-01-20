// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {
struct ConvBiasActivationFusion: public ngraph::pass::GraphRewrite {
    ConvBiasActivationFusion();
};
}  // namespace pass
}  // namespace ArmPlugin
