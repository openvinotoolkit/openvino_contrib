// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <string>

#include "openvino/core/node_output.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/shape.hpp"

namespace ov {
namespace metal_plugin {

// Convert OV element type to MPSGraph data type
MPSDataType to_mps_type(const ov::element::Type& et);

// Convert OV shape to MPSShape (NSArray<NSNumber*>*)
MPSShape* to_mps_shape(const ov::Shape& shape);

}  // namespace metal_plugin
}  // namespace ov
