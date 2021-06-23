// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/pass.hpp>

namespace ngraph::pass {

class FullyConnectedTransformation : public ngraph::pass::FunctionPass {
 public:
  NGRAPH_RTTI_DECLARATION;
  bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};

} // namespace ngraph::pass
