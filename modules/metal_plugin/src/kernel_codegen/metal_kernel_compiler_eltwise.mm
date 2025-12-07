// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <Metal/Metal.h>

#include "kernel_codegen/metal_kernel_compiler.hpp"

#include "mlir_codegen/codegen_common.hpp"
#include "runtime/metal_dtype.hpp"

namespace ov {
namespace metal_plugin {

namespace {
inline EltwiseCodegenDesc make_desc(const KernelOp& op) {
    EltwiseCodegenDesc desc;
    desc.kind = op.kind;
    desc.eltwise_kind = op.kind;
    if (op.output) {
        desc.element_type = op.output->dtype.ov_type;
    }
    desc.is_broadcast = op.is_broadcast;
    desc.out_shape = op.out_shape.empty() && op.output ? op.output->shape : op.out_shape;
    desc.stride0 = op.stride0;
    desc.stride1 = op.stride1;
    return desc;
}
}  // namespace

id<MTLComputePipelineState> MetalKernelCompiler::compile_add_kernel(const KernelOp& op, std::string& log) {
    auto desc = make_desc(op);
    auto source = generate_msl_from_mlir(nullptr, desc);
    return compile_msl_from_source(source, "eltwise_kernel", log);
}

id<MTLComputePipelineState> MetalKernelCompiler::compile_sub_kernel(const KernelOp& op, std::string& log) {
    auto desc = make_desc(op);
    auto source = generate_msl_from_mlir(nullptr, desc);
    return compile_msl_from_source(source, "eltwise_kernel", log);
}

id<MTLComputePipelineState> MetalKernelCompiler::compile_mul_kernel(const KernelOp& op, std::string& log) {
    auto desc = make_desc(op);
    auto source = generate_msl_from_mlir(nullptr, desc);
    return compile_msl_from_source(source, "eltwise_kernel", log);
}

id<MTLComputePipelineState> MetalKernelCompiler::compile_div_kernel(const KernelOp& op, std::string& log) {
    auto desc = make_desc(op);
    if (op.output) {
        desc.num_elements = static_cast<uint32_t>(std::accumulate(op.output->shape.begin(),
                                                                  op.output->shape.end(),
                                                                  int64_t{1},
                                                                  std::multiplies<int64_t>()));
    }
    if (desc.num_elements == 0 && !op.out_shape.empty()) {
        desc.num_elements = static_cast<uint32_t>(std::accumulate(op.out_shape.begin(),
                                                                  op.out_shape.end(),
                                                                  int64_t{1},
                                                                  std::multiplies<int64_t>()));
    }
    auto source = generate_msl_from_mlir(nullptr, desc);
    return compile_msl_from_source(source, "eltwise_kernel", log);
}

id<MTLComputePipelineState> MetalKernelCompiler::compile_mod_kernel(const KernelOp& op, std::string& log) {
    auto desc = make_desc(op);
    auto source = generate_msl_for_eltwise(desc, nullptr);
    return compile_msl_from_source(source, "eltwise_kernel", log);
}

id<MTLComputePipelineState> MetalKernelCompiler::compile_floor_mod_kernel(const KernelOp& op, std::string& log) {
    auto desc = make_desc(op);
    auto source = generate_msl_for_eltwise(desc, nullptr);
    return compile_msl_from_source(source, "eltwise_kernel", log);
}

id<MTLComputePipelineState> MetalKernelCompiler::compile_pow_kernel(const KernelOp& op, std::string& log) {
    auto desc = make_desc(op);
    auto source = generate_msl_from_mlir(nullptr, desc);
    return compile_msl_from_source(source, "eltwise_kernel", log);
}

}  // namespace metal_plugin
}  // namespace ov
