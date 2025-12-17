// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "runtime/metal_op_matmul.hpp"

#include "openvino/core/shape_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "kernel_codegen/metal_kernel_compiler.hpp"
#include "runtime/metal_logger.hpp"

namespace ov {
namespace metal_plugin {

namespace {
inline id<MTLBuffer> to_mtl(const MetalBuffer& buf) {
    return (__bridge id<MTLBuffer>)buf.buffer;
}

// Flatten arbitrary rank (2–4) shapes into [batch, M, K] or [batch, K, N]
std::vector<int64_t> flatten_to_3d(const ov::Shape& s) {
    OPENVINO_ASSERT(!s.empty(), "MatMul: empty shape");
    OPENVINO_ASSERT(s.size() >= 2 && s.size() <= 4, "MatMul supports ranks 2–4");
    int64_t batch = 1;
    for (size_t i = 0; i + 2 < s.size(); ++i) batch *= static_cast<int64_t>(s[i]);
    int64_t m = static_cast<int64_t>(s[s.size() - 2]);
    int64_t k = static_cast<int64_t>(s[s.size() - 1]);
    return {batch, m, k};
}

}  // namespace

MetalMatMulOp::MetalMatMulOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "MatMul",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue),
      m_node(node) {
    fill_desc_from_node(node);
}

void MetalMatMulOp::fill_desc_from_node(const std::shared_ptr<const ov::Node>& node) {
    auto mm0 = ov::as_type_ptr<const ov::op::v0::MatMul>(node);
    OPENVINO_ASSERT(mm0, "MetalMatMulOp expects MatMul node");
    bool ta = mm0->get_transpose_a();
    bool tb = mm0->get_transpose_b();

    m_shape_a = node->get_input_shape(0);
    m_shape_b = node->get_input_shape(1);
    OPENVINO_ASSERT(!m_shape_a.empty() && !m_shape_b.empty(), "MatMul: static shapes required");

    auto a3 = flatten_to_3d(m_shape_a);
    auto b3 = flatten_to_3d(m_shape_b);

    const int64_t batch_a = a3[0];
    const int64_t batch_b = b3[0];
    const int64_t batch = std::max(batch_a, batch_b);
    OPENVINO_ASSERT(batch_a == batch_b || batch_a == 1 || batch_b == 1,
                    "MatMul: incompatible batch broadcast (", batch_a, " vs ", batch_b, ")");

    const int64_t M = ta ? a3[2] : a3[1];
    const int64_t K_a = ta ? a3[1] : a3[2];
    int64_t K_b = tb ? b3[2] : b3[1];
    int64_t N = tb ? b3[1] : b3[2];
    bool b_is_nk_layout = tb;
    if (!tb && K_b != K_a && b3[2] == K_a) {
        // Support pre-transposed weights [N, K].
        K_b = b3[2];
        N = b3[1];
        b_is_nk_layout = true;
    }
    OPENVINO_ASSERT(K_a == K_b, "MatMul: K mismatch (", K_a, " vs ", K_b, ")");

    m_element_type = node->get_output_element_type(0);
    OPENVINO_ASSERT(m_element_type == ov::element::f32 || m_element_type == ov::element::f16,
                    "MetalMatMulOp supports only f16/f32");

    m_desc.kind = KernelOpKind::MatMul;
    m_desc.dtype = resolve_metal_dtype(m_element_type);
    m_desc.element_type = static_cast<uint32_t>(static_cast<ov::element::Type_t>(m_element_type));
    m_desc.M = M;
    m_desc.N = N;
    m_desc.K = K_a;
    m_desc.batch = batch;
    m_desc.batch_a = batch_a;
    m_desc.batch_b = batch_b;
    m_desc.a_transpose = ta;
    m_desc.b_transpose = tb || b_is_nk_layout;
    m_desc.b_is_nk_layout = b_is_nk_layout;
}

void MetalMatMulOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
    auto maybe_upload_const = [&](size_t idx, MetalTensor& tgt) {
        auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(m_node->get_input_node_shared_ptr(idx));
        if (!c)
            return;
        const auto cet = c->get_element_type();
        OPENVINO_ASSERT(cet == m_element_type,
                        "MatMul const type mismatch: expected ",
                        m_element_type.get_type_name(),
                        " got ",
                        cet.get_type_name());
        const size_t bytes = c->get_byte_size();
        tgt.buf = buffer_manager->allocate(bytes,
                                           cet,
                                           /*persistent=*/true,
                                           /*storageModePrivate=*/true);
        buffer_manager->upload(tgt.buf, c->get_data_ptr(), bytes);
        tgt.shape = c->get_shape();
        tgt.expected_type = cet;
    };
    if (m_node) {
        maybe_upload_const(0, m_constA);
        maybe_upload_const(1, m_constB);
    }

    MetalKernelCompiler compiler(m_device ? m_device : (id<MTLDevice>)buffer_manager->device());
    std::string log;
    m_pipeline = compiler.compile_matmul_kernel(m_desc, log);
    OPENVINO_ASSERT(m_pipeline, "MetalMatMulOp: failed to compile matmul kernel: ", log);
}

void MetalMatMulOp::execute() {
    OPENVINO_ASSERT(inputs().size() >= 2, "MatMul: missing inputs");
    MetalTensor* A = inputs()[0] ? inputs()[0] : (m_constA.buf.valid() ? &m_constA : nullptr);
    MetalTensor* B = inputs()[1] ? inputs()[1] : (m_constB.buf.valid() ? &m_constB : nullptr);
    OPENVINO_ASSERT(A && A->buf.valid(), "MatMul: input A is null");
    OPENVINO_ASSERT(B && B->buf.valid(), "MatMul: input B is null");

    MetalTensor& C = require_output();
    if (C.shape.empty()) {
        C.shape = output_shape();
        if (C.shape.empty()) {
            // Derive from static input shapes.
            const int64_t batch = m_desc.batch;
            if (batch == 1)
                C.shape = {static_cast<size_t>(m_desc.M), static_cast<size_t>(m_desc.N)};
            else
                C.shape = {static_cast<size_t>(batch),
                           static_cast<size_t>(m_desc.M),
                           static_cast<size_t>(m_desc.N)};
        }
    }
    if (!C.buf.valid()) {
        size_t bytes = ov::shape_size(C.shape) * m_element_type.size();
        C.buf = buffer_manager()->allocate(bytes, m_element_type, /*persistent=*/false, /*storageModePrivate=*/true);
    }
    C.expected_type = m_element_type;

    const size_t elem_sz = m_element_type.size();
    const size_t need_a = ov::shape_size(A->shape.empty() ? m_shape_a : A->shape) * elem_sz;
    const size_t need_b = ov::shape_size(B->shape.empty() ? m_shape_b : B->shape) * elem_sz;
    const size_t need_c = ov::shape_size(C.shape) * elem_sz;
    OPENVINO_ASSERT(A->buf.size >= need_a, "MatMul: A buffer too small");
    OPENVINO_ASSERT(B->buf.size >= need_b, "MatMul: B buffer too small");
    OPENVINO_ASSERT(C.buf.size >= need_c, "MatMul: C buffer too small");

    if (!m_queue) {
        m_queue = [m_device newCommandQueue];
    }
    id<MTLCommandBuffer> cb = [m_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m_pipeline];
    [enc setBuffer:to_mtl(A->buf) offset:0 atIndex:0];
    [enc setBuffer:to_mtl(B->buf) offset:0 atIndex:1];
    [enc setBuffer:to_mtl(C.buf) offset:0 atIndex:2];

    const uint32_t total = static_cast<uint32_t>(m_desc.batch * m_desc.M * m_desc.N);
    const NSUInteger threads_per_tg = 256;
    MTLSize grid = MTLSizeMake(total, 1, 1);
    MTLSize tg = MTLSizeMake(threads_per_tg, 1, 1);

    start_profiling();
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    stop_profiling_ms();
}

}  // namespace metal_plugin
}  // namespace ov
