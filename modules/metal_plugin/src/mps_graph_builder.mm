// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <MetalPerformanceShadersGraph/MPSGraphConvolutionOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphMatrixMultiplicationOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphTensorData.h>

#include "mps_graph_builder.hpp"

#include <unordered_map>

#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/core/validation_util.hpp"

namespace ov {
namespace metal_plugin {
namespace {

std::string make_key(const ov::Output<const ov::Node>& output) {
    return output.get_node_shared_ptr()->get_friendly_name() + ":" + std::to_string(output.get_index());
}

MPSDataType to_mps_type(const ov::element::Type& et) {
    switch (et) {
    case ov::element::f32:
        return MPSDataTypeFloat32;
    case ov::element::f16:
        return MPSDataTypeFloat16;
    default:
        OPENVINO_THROW("METAL plugin: unsupported element type for MPSGraph: ", et);  // NOLINT
    }
}

MPSShape* to_mps_shape(const ov::Shape& shape) {
    NSMutableArray<NSNumber*>* arr = [NSMutableArray arrayWithCapacity:shape.size()];
    for (auto dim : shape) {
        [arr addObject:@(dim)];
    }
    // MPSShape is typedef NSArray<NSNumber*>*
    return [NSArray arrayWithArray:arr];
}

}  // namespace

MPSGraphBuildResult build_mps_graph(const std::shared_ptr<const ov::Model>& model, GraphLayout layout) {
    OPENVINO_ASSERT(model, "Model is null");

    MPSGraph* graph = [[MPSGraph alloc] init];

    std::unordered_map<std::string, MPSGraphTensor*> tensor_map;
    std::vector<void*> inputs;
    std::vector<void*> outputs;

    // Process nodes in topological order
    for (const auto& node : model->get_ordered_ops()) {
        if (auto p = ov::as_type_ptr<const ov::op::v0::Parameter>(node)) {
            auto shape = to_mps_shape(p->get_shape());
            auto dtype = to_mps_type(p->get_element_type());
            auto placeholder = [graph placeholderWithShape:shape dataType:dtype name:nil];
            tensor_map[make_key(p->output(0))] = placeholder;
            inputs.push_back(placeholder);
        } else if (auto c = ov::as_type_ptr<const ov::op::v0::Constant>(node)) {
            auto shape = to_mps_shape(c->get_shape());
            auto dtype = to_mps_type(c->get_element_type());
            NSData* data = [NSData dataWithBytes:c->get_data_ptr() length:c->get_byte_size()];
            auto tensor = [graph constantWithData:data shape:shape dataType:dtype];
            tensor_map[make_key(c->output(0))] = tensor;
        } else if (ov::as_type_ptr<const ov::op::v1::Add>(node)) {
            auto a = node->input_value(0);
            auto b = node->input_value(1);
            auto it_a = tensor_map.find(make_key(a));
            auto it_b = tensor_map.find(make_key(b));
            OPENVINO_ASSERT(it_a != tensor_map.end(), "Add: missing lhs tensor");
            OPENVINO_ASSERT(it_b != tensor_map.end(), "Add: missing rhs tensor");
            if (a.get_shape() != b.get_shape()) {
                OPENVINO_THROW("Add: broadcasting is not implemented; shapes differ");
            }
            auto res = [graph additionWithPrimaryTensor:it_a->second secondaryTensor:it_b->second name:nil];
            tensor_map[make_key(node->output(0))] = res;
        } else if (ov::as_type_ptr<const ov::op::v0::Relu>(node)) {
            auto inp = node->input_value(0);
            auto it = tensor_map.find(make_key(inp));
            OPENVINO_ASSERT(it != tensor_map.end(), "Relu: missing input tensor");
            auto res = [graph reLUWithTensor:it->second name:nil];
            tensor_map[make_key(node->output(0))] = res;
        } else if (auto mm0 = ov::as_type_ptr<const ov::op::v0::MatMul>(node)) {
            bool transpose_a = mm0->get_transpose_a();
            bool transpose_b = mm0->get_transpose_b();
            auto a = node->input_value(0);
            auto b = node->input_value(1);
            auto it_a = tensor_map.find(make_key(a));
            auto it_b = tensor_map.find(make_key(b));
            OPENVINO_ASSERT(it_a != tensor_map.end(), "MatMul: missing A tensor");
            OPENVINO_ASSERT(it_b != tensor_map.end(), "MatMul: missing B tensor");
            auto rank_a = a.get_partial_shape().rank();
            auto rank_b = b.get_partial_shape().rank();
            if (!(rank_a.is_static() && rank_b.is_static())) {
                OPENVINO_THROW("MatMul: dynamic ranks are not supported yet");
            }
            auto ra = rank_a.get_length();
            auto rb = rank_b.get_length();
            if (!((ra == 2 && rb == 2) || (ra == 3 && rb == 3))) {
                OPENVINO_THROW("MatMul: only 2D or 3D batched matmul is supported");
            }
            if (transpose_a || transpose_b) {
                OPENVINO_THROW("MatMul: transpose flags not supported in current MPSGraph path");
            }
            auto res = [graph matrixMultiplicationWithPrimaryTensor:it_a->second
                                                    secondaryTensor:it_b->second
                                                               name:nil];
            tensor_map[make_key(node->output(0))] = res;
        } else if (auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(node)) {
            const auto& strides = conv->get_strides();
            const auto& pads_begin = conv->get_pads_begin();
            const auto& pads_end = conv->get_pads_end();
            const auto& dilations = conv->get_dilations();
            if (strides.size() != 2 || pads_begin.size() != 2 || pads_end.size() != 2 || dilations.size() != 2) {
                OPENVINO_THROW("Convolution: unsupported stride/pads/dilations configuration");
            }
            auto data = node->input_value(0);
            auto weights = node->input_value(1);
            auto it_d = tensor_map.find(make_key(data));
            auto it_w = tensor_map.find(make_key(weights));
            OPENVINO_ASSERT(it_d != tensor_map.end(), "Convolution: missing data tensor");
            OPENVINO_ASSERT(it_w != tensor_map.end(), "Convolution: missing weights tensor");
            auto* src = it_d->second;
            auto* wts = it_w->second;
            MPSGraphConvolution2DOpDescriptor* desc =
                [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:(NSUInteger)strides[1]
                                                               strideInY:(NSUInteger)strides[0]
                                                       dilationRateInX:(NSUInteger)dilations[1]
                                                       dilationRateInY:(NSUInteger)dilations[0]
                                                                groups:1
                                                           paddingLeft:(NSUInteger)pads_begin[1]
                                                          paddingRight:(NSUInteger)pads_end[1]
                                                            paddingTop:(NSUInteger)pads_begin[0]
                                                         paddingBottom:(NSUInteger)pads_end[0]
                                                         paddingStyle:MPSGraphPaddingStyleExplicit
                                                             dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                                                          weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
            auto res = [graph convolution2DWithSourceTensor:src
                                             weightsTensor:wts
                                                descriptor:desc
                                                      name:nil];
            tensor_map[make_key(node->output(0))] = res;
        } else if (ov::as_type_ptr<const ov::op::v0::Result>(node)) {
            continue;
        } else {
            OPENVINO_THROW("METAL plugin: lowering for op is not implemented: ", node->get_friendly_name());
        }
    }

    // Collect outputs (Result nodes point to actual output tensors)
    outputs.reserve(model->outputs().size());
    for (const auto& output : model->outputs()) {
        auto src = output.get_node_shared_ptr();
        if (auto res = ov::as_type_ptr<const ov::op::v0::Result>(src)) {
            auto inp = res->input_value(0);
            auto it = tensor_map.find(make_key(inp));
            OPENVINO_ASSERT(it != tensor_map.end(), "Missing tensor for result input " + inp.get_node()->get_friendly_name());
            outputs.push_back(it->second);
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

void mps_execute(const std::shared_ptr<void>& graph_sp,
                 const std::vector<void*>& input_tensors,
                 const std::vector<void*>& output_tensors,
                 const std::vector<ov::Tensor>& inputs,
                 std::vector<ov::Tensor>& outputs) {
    auto graph = (__bridge MPSGraph*)graph_sp.get();
    OPENVINO_ASSERT(graph, "MPSGraph is null");
    OPENVINO_ASSERT(inputs.size() == input_tensors.size(), "Input tensors count mismatch");
    OPENVINO_ASSERT(outputs.size() == output_tensors.size(), "Output tensors count mismatch");

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    OPENVINO_ASSERT(device, "Failed to get default Metal device");
    id<MTLCommandQueue> queue = [device newCommandQueue];
    OPENVINO_ASSERT(queue, "Failed to create Metal command queue");

    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [NSMutableDictionary dictionary];

    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& tin = inputs[i];
        auto buf = [device newBufferWithLength:tin.get_byte_size() options:MTLResourceStorageModeShared];
        OPENVINO_ASSERT(buf, "Failed to allocate Metal buffer for input");
        std::memcpy([buf contents], tin.data(), tin.get_byte_size());

        auto shape = to_mps_shape(tin.get_shape());
        auto dtype = to_mps_type(tin.get_element_type());
        MPSGraphTensorData* td = [[MPSGraphTensorData alloc] initWithMTLBuffer:buf shape:shape dataType:dtype];
        [feeds setObject:td forKey:(__bridge MPSGraphTensor*)input_tensors[i]];
        [td release];
        [buf release];
    }

    NSMutableArray<MPSGraphTensor*>* targets = [NSMutableArray arrayWithCapacity:output_tensors.size()];
    for (auto* t : output_tensors) {
        [targets addObject:(__bridge MPSGraphTensor*)t];
    }

    MPSGraphTensorDataDictionary* results = [graph runWithMTLCommandQueue:queue
                                                                    feeds:feeds
                                                            targetTensors:targets
                                                         targetOperations:nil];

    for (size_t i = 0; i < outputs.size(); ++i) {
        auto* t = (__bridge MPSGraphTensor*)output_tensors[i];
        MPSGraphTensorData* td = [results objectForKey:t];
        OPENVINO_ASSERT(td, "Missing output tensor data");
        MPSNDArray* nda = [td mpsndarray];
        OPENVINO_ASSERT(nda, "Failed to obtain MPSNDArray for output");
        auto& tout = outputs[i];
        size_t expected_bytes = [nda dataTypeSize];
        for (NSUInteger d = 0; d < [nda numberOfDimensions]; ++d) {
            expected_bytes *= [nda lengthOfDimension:d];
        }
        OPENVINO_ASSERT(tout.get_byte_size() == expected_bytes, "Output size mismatch");
        [nda readBytes:tout.data() strideBytes:nullptr];
    }

    [queue release];
}

}  // namespace metal_plugin
}  // namespace ov
