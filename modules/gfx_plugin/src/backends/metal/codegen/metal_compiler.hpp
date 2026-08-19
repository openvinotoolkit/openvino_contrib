// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#import <Metal/Metal.h>

#include <string>

namespace ov {
namespace gfx_plugin {

class MetalKernelCompiler {
public:
    explicit MetalKernelCompiler(id<MTLDevice> device) : m_device(device) {}

    id<MTLComputePipelineState> compile_msl_from_source(const std::string& source,
                                                        const char* entry_point,
                                                        std::string& log);

private:
    id<MTLDevice> m_device = nil;
};

}  // namespace gfx_plugin
}  // namespace ov
