// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <Foundation/Foundation.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "ops/common.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace metal_plugin {

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

}  // namespace metal_plugin
}  // namespace ov
