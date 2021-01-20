// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {

struct ConvertComparison: public ngraph::pass::GraphRewrite {
    ConvertComparison();
};
}  // namespace pass
}  // namespace ArmPlugin
