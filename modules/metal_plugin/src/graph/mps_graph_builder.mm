// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <Foundation/Foundation.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "graph/mps_graph_builder.hpp"

#include <unordered_map>

#include "graph/mps_node_context.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "ops/common.hpp"
#include "ops/convolution.hpp"
#include "ops/elementwise.hpp"
#include "ops/matmul.hpp"
#include "ops/pooling.hpp"
#include "ops/softmax.hpp"
#include "ops/batchnorm.hpp"
#include "transforms/pipeline.hpp"

namespace ov {
namespace metal_plugin {

MPSGraphBuildResult build_mps_graph(const std::shared_ptr<const ov::Model>& model, GraphLayout layout) {
    OPENVINO_ASSERT(model, "Model is null");
    auto transformed_model = transforms::run_pipeline(model);

    MPSGraph* graph = [[MPSGraph alloc] init];

    std::vector<std::unique_ptr<MetalNode>> nodes;
    std::unordered_map<const ov::Node*, MetalNode*> node_map;
    NodeContext ctx{graph, nodes, node_map, Layout::NCHW};

    std::vector<void*> inputs;
    std::vector<void*> outputs;

    for (const auto& node : transformed_model->get_ordered_ops()) {
        if (auto p = ov::as_type_ptr<const ov::op::v0::Parameter>(node)) {
            auto shape = NodeContext::to_mps_shape(p->get_shape());
            auto dtype = to_mps_type(p->get_element_type());
            auto placeholder = [graph placeholderWithShape:shape dataType:dtype name:nil];

            MetalNode* mn = ctx.create_node(MetalOpType::Parameter, p, {});
            mn->output_desc.shape = p->get_shape();
            mn->output_desc.element_type = p->get_element_type();
            mn->output_desc.layout = Layout::NCHW;
            mn->mps_tensor = placeholder;

            inputs.push_back(placeholder);
        } else if (auto c = ov::as_type_ptr<const ov::op::v0::Constant>(node)) {
            ov::Shape const_shape = c->get_shape();
            if (const_shape.empty()) {
                const_shape = {1};  // MPSGraph requires ranked shape
            }
            auto shape = NodeContext::to_mps_shape(const_shape);
            auto dtype = to_mps_type(c->get_element_type());
            NSData* data = [NSData dataWithBytes:c->get_data_ptr() length:c->get_byte_size()];
            auto tensor = [graph constantWithData:data shape:shape dataType:dtype];

            MetalNode* mn = ctx.create_node(MetalOpType::Constant, c, {});
            mn->output_desc.shape = c->get_shape();
            mn->output_desc.element_type = c->get_element_type();
            mn->output_desc.layout = Layout::NCHW;
            mn->mps_tensor = tensor;
        } else if (auto add = ov::as_type_ptr<const ov::op::v1::Add>(node)) {
            ops::build_add(ctx, *add);
        } else if (auto relu = ov::as_type_ptr<const ov::op::v0::Relu>(node)) {
            ops::build_relu(ctx, *relu);
        } else if (auto mm = ov::as_type_ptr<const ov::op::v0::MatMul>(node)) {
            ops::build_matmul(ctx, *mm);
        } else if (auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(node)) {
            ops::build_convolution(ctx, *conv);
        } else if (auto maxp = ov::as_type_ptr<const ov::op::v1::MaxPool>(node)) {
            ops::build_max_pool(ctx, *maxp);
        } else if (auto avgp = ov::as_type_ptr<const ov::op::v1::AvgPool>(node)) {
            ops::build_avg_pool(ctx, *avgp);
        } else if (auto sm = ov::as_type_ptr<const ov::op::v1::Softmax>(node)) {
            ops::build_softmax(ctx, *sm);
        } else if (ov::as_type_ptr<const ov::op::v0::Result>(node)) {
            continue;
        } else if (auto bn5 = ov::as_type_ptr<const ov::op::v5::BatchNormInference>(node)) {
            ops::build_batch_norm(ctx, *bn5);
        } else if (auto bn0 = ov::as_type_ptr<const ov::op::v0::BatchNormInference>(node)) {
            ops::build_batch_norm(ctx, *bn0);
        } else {
            OPENVINO_THROW("METAL plugin: lowering for op is not implemented: ", node->get_friendly_name());
        }
    }

    outputs.reserve(transformed_model->outputs().size());
    for (const auto& output : transformed_model->outputs()) {
        auto src = output.get_node_shared_ptr();
        if (auto res = ov::as_type_ptr<const ov::op::v0::Result>(src)) {
            auto inp = res->input_value(0);
            auto* producer = ctx.get_node(inp.get_node());
            OPENVINO_ASSERT(producer && producer->mps_tensor, "Missing tensor for result input");
            outputs.push_back(producer->mps_tensor);
        } else {
            OPENVINO_THROW("METAL plugin: unexpected output node type");
        }
    }

    MPSGraphBuildResult result;
    CFRetain((__bridge CFTypeRef)graph);
    result.graph = std::shared_ptr<void>((__bridge void*)graph, [](void* p) {
        if (p) {
            CFRelease(p);
        }
    });
    result.input_tensors = std::move(inputs);
    result.output_tensors = std::move(outputs);
    result.internal_layout = layout;

    [graph release];

    return result;
}

}  // namespace metal_plugin
}  // namespace ov
