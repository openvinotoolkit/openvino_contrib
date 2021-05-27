// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_graph_transformer.hpp"

#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/init_node_info.hpp>

#include "cuda/cuda_config.hpp"
#include "cuda_pattern_transformation.hpp"

using namespace CUDAPlugin;

std::shared_ptr<ngraph::Function> GraphTransformer::transform(
    const std::shared_ptr<const ngraph::Function> &function,
    const std::map<std::string, std::string> &) const {
  auto transformed_function = ngraph::clone_function(*function);

  ngraph::pass::Manager manager;

  manager.register_pass<ngraph::pass::InitNodeInfo>();
  manager.register_pass<ngraph::pass::CommonOptimizations>();

  manager.run_passes(transformed_function);

  return transformed_function;
}
