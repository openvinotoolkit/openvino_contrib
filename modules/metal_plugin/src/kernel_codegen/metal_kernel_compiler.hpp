// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#import <Metal/Metal.h>

#include <string>

#include "kernel_ir/kernel_ir_common.hpp"

namespace ov {
namespace metal_plugin {

class MetalKernelCompiler {
public:
    explicit MetalKernelCompiler(id<MTLDevice> device) : m_device(device) {}

    id<MTLComputePipelineState> compile_add_kernel(const KernelOp& op, std::string& log);
    id<MTLComputePipelineState> compile_sub_kernel(const KernelOp& op, std::string& log);
    id<MTLComputePipelineState> compile_div_kernel(const KernelOp& op, std::string& log);
    id<MTLComputePipelineState> compile_mod_kernel(const KernelOp& op, std::string& log);
    id<MTLComputePipelineState> compile_floor_mod_kernel(const KernelOp& op, std::string& log);
    id<MTLComputePipelineState> compile_pow_kernel(const KernelOp& op, std::string& log);
    id<MTLComputePipelineState> compile_matmul_kernel(const KernelOp& op, std::string& log);
    id<MTLComputePipelineState> compile_unary_kernel(const KernelOp& op, std::string& log);
    id<MTLComputePipelineState> compile_softmax_kernel(const KernelOp& op, std::string& log);
    id<MTLComputePipelineState> compile_maxpool2d_kernel(const KernelOp& op, std::string& log);
    id<MTLComputePipelineState> compile_avgpool2d_kernel(const KernelOp& op, std::string& log);
    id<MTLComputePipelineState> compile_conv2d_kernel(const KernelOp& op, std::string& log);
    id<MTLComputePipelineState> compile_conv3d_kernel(const KernelOp& op, std::string& log);
    id<MTLComputePipelineState> compile_batchnorm2d_kernel(const KernelOp& op, std::string& log);
    id<MTLComputePipelineState> compile_mul_kernel(const KernelOp& op, std::string& log);
    id<MTLComputePipelineState> compile_slice_kernel(const KernelOp& op, std::string& log);
    id<MTLComputePipelineState> compile_split_kernel(const KernelOp& op, std::string& log);
    id<MTLComputePipelineState> compile_msl_from_source(const std::string& source,
                                                        const char* entry_point,
                                                        std::string& log);

private:
    id<MTLDevice> m_device = nil;
};

}  // namespace metal_plugin
}  // namespace ov
