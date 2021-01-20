// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {
struct ConvertReduceSingleAxis : public ngraph::pass::GraphRewrite {
    ConvertReduceSingleAxis();
};
}  // namespace pass
}  // namespace ArmPlugin
