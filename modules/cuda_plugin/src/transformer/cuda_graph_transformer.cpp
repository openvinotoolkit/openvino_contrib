// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_graph_transformer.hpp"

#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/init_node_info.hpp>
#include <transformer/fuse_conv2d_biasadd_activation.hpp>

#include "cuda/cuda_config.hpp"
#include "cuda_fullyconnected_transformation.hpp"

using namespace CUDAPlugin;

std::shared_ptr<ngraph::Function> GraphTransformer::transform(
    const std::shared_ptr<const ngraph::Function> &function,
    const std::map<std::string, std::string> &) const {
  auto transformed_function = ngraph::clone_function(*function);

  ngraph::pass::Manager manager;

  [[maybe_unused]] const auto& originOps = function->get_ordered_ops();

  manager.register_pass<ngraph::pass::InitNodeInfo>();
  // TODO: enable whenever Conv2DBiasAdd Op implementation available
  // manager.register_pass<ngraph::pass::CudaFuseConv2DBiasAddActivation>();
  manager.register_pass<ngraph::pass::CommonOptimizations>();
  manager.register_pass<ngraph::pass::FullyConnectedTransformation>();

  manager.run_passes(transformed_function);

  [[maybe_unused]] const auto& transformedOps = transformed_function->get_ordered_ops();

  return transformed_function;
}
