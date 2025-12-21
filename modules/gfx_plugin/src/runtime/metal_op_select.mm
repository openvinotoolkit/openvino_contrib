// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "runtime/metal_op_select.hpp"

#include <limits>

#include "openvino/core/except.hpp"
#include "openvino/op/parameter.hpp"
#include "runtime/metal_logger.hpp"
#include "runtime/metal_op_utils.hpp"
#include "mlir_builder.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

namespace ov {
namespace gfx_plugin {

namespace {
inline id<MTLBuffer> to_mtl(const MetalBuffer& buf) {
    return (__bridge id<MTLBuffer>)buf.buffer;
}
}  // namespace

MetalSelectOp::MetalSelectOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "Select",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_node(node),
      m_element_type(node->get_output_element_type(0)),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {}

void MetalSelectOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalSelectOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }

    MetalKernelCompiler compiler(m_device ? m_device : (id<MTLDevice>)buffer_manager->device());
    std::string log;
    mlir::MLIRContext ctx;
    auto module = build_mlir_select_from_model(make_single_op_model(m_node), ctx);
    auto source = generate_msl_for_select(module, m_element_type);
    m_pipeline = compiler.compile_msl_from_source(source, "select_kernel", log);
    OPENVINO_ASSERT(m_pipeline, "MetalSelectOp: failed to compile select kernel: ", log);

    m_num_elems = static_cast<uint32_t>(ov::shape_size(output_shape()));
    MetalOp::compile(buffer_manager);
}

void MetalSelectOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(inputs().size() >= 3, "Select: requires 3 inputs (cond, then, else)");
    MetalTensor* cond = inputs()[0];
    MetalTensor* tval = inputs()[1];
    MetalTensor* fval = inputs()[2];
    OPENVINO_ASSERT(cond && cond->buf.valid(), "Select: cond buffer null");
    OPENVINO_ASSERT(tval && tval->buf.valid(), "Select: true buffer null");
    OPENVINO_ASSERT(fval && fval->buf.valid(), "Select: false buffer null");

    MetalTensor& out = require_output();
    ov::Shape out_shape = !out.shape.empty() ? out.shape : output_shape();
    if (out_shape.empty() && m_node->get_output_partial_shape(0).is_static()) {
        out_shape = m_node->get_output_shape(0);
    }
    OPENVINO_ASSERT(!out_shape.empty(), "Select: output shape unknown");
    const size_t bytes = m_element_type.size() * ov::shape_size(out_shape);
    if (!out.buf.valid() || out.buf.size < bytes) {
        out.buf = buffer_manager()->allocate(bytes, m_element_type, /*persistent=*/false, out.prefer_private);
    }
    out.expected_type = m_element_type;
    out.shape = out_shape;

    id<MTLCommandBuffer> cb = static_cast<id<MTLCommandBuffer>>(cmd_buf_handle);
    OPENVINO_ASSERT(cb, "MetalSelectOp: command buffer is null");
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m_pipeline];

    // Compute broadcast strides for cond, tval, fval to output.
    auto make_stride = [](const ov::Shape& shp) {
        std::vector<int> st(shp.size(), 1);
        if (shp.empty()) return st;
        for (int i = static_cast<int>(shp.size()) - 2; i >= 0; --i) {
            st[i] = st[i + 1] * static_cast<int>(shp[i + 1]);
        }
        return st;
    };
    const size_t rank = out_shape.empty() ? 1 : out_shape.size();
    std::vector<int> out_dims(rank, 1);
    for (size_t i = 0; i < out_shape.size(); ++i) out_dims[i] = static_cast<int>(out_shape[i]);

    auto norm_shape = [&](const ov::Shape& s) {
        std::vector<size_t> r(rank, 1);
        if (s.empty()) return r;
        size_t off = rank - s.size();
        for (size_t i = 0; i < s.size(); ++i) r[off + i] = s[i];
        return r;
    };

    auto cond_norm = norm_shape(cond->shape);
    auto a_norm = norm_shape(tval->shape);
    auto b_norm = norm_shape(fval->shape);

    auto elem_size_for = [](const MetalTensor* t) -> size_t {
        if (!t) return 0;
        const auto et = t->expected_type == ov::element::dynamic ? t->buf.type : t->expected_type;
        return et.size();
    };
    auto shape_elems = [](const std::vector<size_t>& s) {
        size_t prod = 1;
        for (auto v : s) prod *= v;
        return prod;
    };
    const size_t cond_bytes = shape_elems(cond_norm) * elem_size_for(cond);
    const size_t a_bytes = shape_elems(a_norm) * elem_size_for(tval);
    const size_t b_bytes = shape_elems(b_norm) * elem_size_for(fval);
    OPENVINO_ASSERT(cond->buf.size >= cond_bytes, "Select: cond buffer too small");
    OPENVINO_ASSERT(tval->buf.size >= a_bytes, "Select: true buffer too small");
    OPENVINO_ASSERT(fval->buf.size >= b_bytes, "Select: false buffer too small");

    auto cond_stride_full = make_stride(cond_norm);
    auto a_stride_full = make_stride(a_norm);
    auto b_stride_full = make_stride(b_norm);

    std::vector<int> stride_c(rank, 0), stride_a(rank, 0), stride_b(rank, 0);
    for (size_t d = 0; d < rank; ++d) {
        stride_c[d] = cond_norm[d] == 1 ? 0 : cond_stride_full[d];
        stride_a[d] = a_norm[d] == 1 ? 0 : a_stride_full[d];
        stride_b[d] = b_norm[d] == 1 ? 0 : b_stride_full[d];
    }

    [enc setBuffer:to_mtl(cond->buf) offset:0 atIndex:0];
    [enc setBuffer:to_mtl(tval->buf) offset:0 atIndex:1];
    [enc setBuffer:to_mtl(fval->buf) offset:0 atIndex:2];
    [enc setBuffer:to_mtl(out.buf) offset:0 atIndex:3];
    uint32_t num = m_num_elems ? m_num_elems : static_cast<uint32_t>(ov::shape_size(out.shape));
    uint32_t r = static_cast<uint32_t>(rank ? rank : 1);
    [enc setBytes:&num length:sizeof(num) atIndex:4];
    [enc setBytes:&r length:sizeof(r) atIndex:5];
    [enc setBytes:out_dims.data() length:out_dims.size() * sizeof(int) atIndex:6];
    [enc setBytes:stride_c.data() length:stride_c.size() * sizeof(int) atIndex:7];
    [enc setBytes:stride_a.data() length:stride_a.size() * sizeof(int) atIndex:8];
    [enc setBytes:stride_b.data() length:stride_b.size() * sizeof(int) atIndex:9];

    if (num == 0) {
        [enc endEncoding];
        return;
    }
    const NSUInteger threads = 64;
    MTLSize grid = MTLSizeMake(num, 1, 1);
    const NSUInteger tg_size = static_cast<NSUInteger>(metal_clamp_tg_size((void*)m_pipeline, threads));
    MTLSize tg = MTLSizeMake(tg_size, 1, 1);
    start_profiling(enc);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    stop_profiling_ms(enc);
    [enc endEncoding];

    out.expected_type = m_element_type;
}

}  // namespace gfx_plugin
}  // namespace ov
