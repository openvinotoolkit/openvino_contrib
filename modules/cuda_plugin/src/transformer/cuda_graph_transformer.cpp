// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_graph_transformer.hpp"

#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/common_optimizations/conv_bias_fusion.hpp>
#include <transformations/convert_precision.hpp>
#include <transformations/init_node_info.hpp>
#include <transformer/fuse_conv_biasadd_activation.hpp>

#include "concat_transformation.hpp"
#include "cuda/cuda_config.hpp"
#include "cuda_fullyconnected_transformation.hpp"

using namespace CUDAPlugin;

std::shared_ptr<ngraph::Function> GraphTransformer::transform(
    const CUDA::Device& device,
    const std::shared_ptr<const ngraph::Function> &function,
    const std::map<std::string, std::string> &) const {
  auto transformed_function = ngraph::clone_function(*function);

  ngraph::pass::Manager manager;

  [[maybe_unused]] const auto& originOps = function->get_ordered_ops();
  [[maybe_unused]] const auto& originOpsSize = originOps.size();

  manager.register_pass<ngraph::pass::InitNodeInfo>();
  manager.register_pass<ngraph::pass::CommonOptimizations>();
  if (!isHalfSupported(device)) {
    manager.register_pass<ngraph::pass::ConvertPrecision>(
        ngraph::element::f16, ngraph::element::f32);
  }
  if (!isInt8Supported(device)) {
    manager.register_pass<ngraph::pass::ConvertPrecision>(
        ngraph::element::i8, isHalfSupported(device) ? ngraph::element::f16 : ngraph::element::f32);
  }
  manager.register_pass<ngraph::pass::CudaFuseConvBiasAddActivation>();
  manager.register_pass<ngraph::pass::CudaFuseConvBackpropDataAdd>();
  manager.register_pass<ngraph::pass::FullyConnectedTransformation>();
  manager.register_pass<ngraph::pass::ConcatTransformation>();

  manager.run_passes(transformed_function);

  [[maybe_unused]] const auto& transformedOps = transformed_function->get_ordered_ops();
  [[maybe_unused]] const auto& transformedOpsSize = transformedOps.size();

  return transformed_function;
}
