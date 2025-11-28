// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <MetalPerformanceShadersGraph/MPSGraphTensorData.h>

#include "runtime/mps_executor.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace metal_plugin {
namespace {

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
    return [NSArray arrayWithArray:arr];
}

}  // namespace

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
