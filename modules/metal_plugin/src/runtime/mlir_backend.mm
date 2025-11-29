// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "runtime/mlir_backend.hpp"

#import <Metal/Metal.h>

#include <numeric>

#include "kernel_codegen/kernel_ir.hpp"
#include "kernel_codegen/metal_kernel_compiler.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace metal_plugin {

class MlirBackend::Impl {
public:
    explicit Impl(const std::shared_ptr<const ov::Model>& model) {
        m_device = MTLCreateSystemDefaultDevice();
        OPENVINO_ASSERT(m_device, "MlirBackend: failed to create Metal device");
        m_queue = [m_device newCommandQueue];
        OPENVINO_ASSERT(m_queue, "MlirBackend: failed to create command queue");
        compile(model);
    }

    void run(const std::vector<ov::Tensor>& inputs, std::vector<ov::Tensor>& outputs) {
        OPENVINO_ASSERT(inputs.size() == 2 && outputs.size() == 1, "MlirBackend: expected 2 inputs and 1 output");
        OPENVINO_ASSERT(m_pipeline, "MlirBackend: pipeline is null");

        id<MTLBuffer> buf0 = [m_device newBufferWithBytes:inputs[0].data()
                                                 length:inputs[0].get_byte_size()
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf1 = [m_device newBufferWithBytes:inputs[1].data()
                                                 length:inputs[1].get_byte_size()
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_out = [m_device newBufferWithLength:outputs[0].get_byte_size()
                                                     options:MTLResourceStorageModeShared];
        OPENVINO_ASSERT(buf0 && buf1 && buf_out, "MlirBackend: failed to allocate buffers");

        id<MTLCommandBuffer> cmd = [m_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:m_pipeline];
        [enc setBuffer:buf0 offset:0 atIndex:0];
        [enc setBuffer:buf1 offset:0 atIndex:1];
        [enc setBuffer:buf_out offset:0 atIndex:2];

        if (m_ir.ops[0].kind == KernelOpKind::MatMul) {
            const NSUInteger M = static_cast<NSUInteger>(m_ir.ops[0].M);
            const NSUInteger N = static_cast<NSUInteger>(m_ir.ops[0].N);
            MTLSize grid = MTLSizeMake(N, M, 1);
            MTLSize tg = MTLSizeMake(8, 8, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        } else {
            size_t elems = outputs[0].get_size();
            const NSUInteger threads_per_tg = 64;
            MTLSize grid = MTLSizeMake(elems, 1, 1);
            MTLSize tg = MTLSizeMake(threads_per_tg, 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        }
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        std::memcpy(outputs[0].data(), [buf_out contents], outputs[0].get_byte_size());

        [buf0 release];
        [buf1 release];
        [buf_out release];
    }

private:
    void compile(const std::shared_ptr<const ov::Model>& model) {
        // TODO: real MLIR lowering. For now, reuse lightweight IR builders.
        MetalKernelCompiler compiler(m_device);
        std::string log;
        try {
            m_ir = build_kernel_ir_for_matmul(model);
            m_pipeline = compiler.compile_matmul_kernel(m_ir.ops[0], log);
        } catch (const ov::Exception&) {
            m_ir = build_kernel_ir_for_add(model);
            m_pipeline = compiler.compile_add_kernel(m_ir.ops[0], log);
        }
    }

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    MetalKernelIR m_ir;
    id<MTLComputePipelineState> m_pipeline = nil;
};

MlirBackend::MlirBackend(const std::shared_ptr<const ov::Model>& model)
    : m_impl(std::make_unique<MlirBackend::Impl>(model)) {}

MlirBackend::~MlirBackend() = default;

void MlirBackend::run(const std::vector<ov::Tensor>& inputs, std::vector<ov::Tensor>& outputs) {
    m_impl->run(inputs, outputs);
}

}  // namespace metal_plugin
}  // namespace ov
