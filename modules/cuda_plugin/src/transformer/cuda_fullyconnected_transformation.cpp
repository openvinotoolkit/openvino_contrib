// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_fullyconnected_transformation.hpp"

#include <exec_graph_info.hpp>
#include <ngraph/op/add.hpp>
#include <ngraph/op/matmul.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>
#include <ops/matmul.hpp>
#include <transformer/nodes/fully_connected.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::FullyConnectedTransformation,
                       "MyFunctionTransformation", 0);

namespace ngraph::pass {

bool FullyConnectedTransformation::run_on_function(
    std::shared_ptr<ngraph::Function> f) {
  bool isGraphUpdated = false;
  // Example transformation code
  const auto nodes = f->get_ordered_ops();
  // Traverse nGraph Function in topological order
  for (int i = 0; i < nodes.size(); ++i) {
    auto& node = nodes[i];
    // Check that input and output shape a fully defined (not dynamic) and
    // number of consumers equal to 1
    if (node->inputs().size() >= 1 && node->outputs().size() >= 1) {
      auto inputNode = node->get_input_node_shared_ptr(0);
      if (ngraph::is_type<ngraph::op::MatMul>(inputNode) &&
          ngraph::is_type<ngraph::op::v1::Add>(node)) {
        // Decompose Divide into Multiply with Power operations
        auto matMulNode = std::dynamic_pointer_cast<ngraph::op::v0::MatMul>(inputNode);
        auto matrixAShape = matMulNode->get_input_shape(0);
        auto matrixBShape = matMulNode->get_input_shape(1);
        CUDAPlugin::MatMulOp::BroadcastToMatrix(matrixAShape);
        CUDAPlugin::MatMulOp::BroadcastToMatrix(matrixBShape);
        const auto matMulBatch = std::max(
            CUDAPlugin::MatMulOp::GetMatrixNumBatches(matrixAShape),
            CUDAPlugin::MatMulOp::GetMatrixNumBatches(matrixBShape));
        auto addNode = node;
        auto constantNode = addNode->get_input_node_shared_ptr(1);
        auto constShape = constantNode->get_output_shape(0);
        CUDAPlugin::MatMulOp::BroadcastToMatrix(constShape);
        const auto constBatch = CUDAPlugin::MatMulOp::GetMatrixNumBatches(constShape);
        if (matMulBatch < constBatch) {
          continue;
        }

        auto fullyConnectedNode = std::make_shared<CUDAPlugin::nodes::FullyConnected>(
            matMulNode->get_input_node_shared_ptr(0),
            matMulNode->get_input_node_shared_ptr(1),
            constantNode,
            matMulNode->get_transpose_a(),
            matMulNode->get_transpose_b());
        fullyConnectedNode->set_friendly_name(addNode->get_friendly_name());
        ngraph::copy_runtime_info({matMulNode, addNode}, fullyConnectedNode);
        const std::string originalLayers =
            matMulNode->get_friendly_name() + "," + addNode->get_friendly_name();
        fullyConnectedNode->get_rt_info()[ExecGraphInfoSerialization::ORIGINAL_NAMES] =
            std::make_shared<ngraph::VariantWrapper<std::string>>(originalLayers);

        auto addOutputNode = addNode->output(0);
        addOutputNode.replace(Output<Node>{fullyConnectedNode});

        isGraphUpdated = true;
      }
    }
  }

  // Return false because we didn't change nGraph Function
  return isGraphUpdated;
}

} // namespace ngraph::pass
