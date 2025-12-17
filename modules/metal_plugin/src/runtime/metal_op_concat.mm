// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "runtime/metal_op_concat.hpp"

#include "openvino/core/shape_util.hpp"
#include "runtime/metal_logger.hpp"

namespace ov {
namespace metal_plugin {

namespace {
inline id<MTLBuffer> to_mtl(const MetalBuffer& buf) {
    return (__bridge id<MTLBuffer>)buf.buffer;
}

size_t element_size(const ov::element::Type& t) {
    return t.size();
}
}  // namespace

MetalConcatOp::MetalConcatOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "Concat",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
    compute_layout(node);
}

void MetalConcatOp::compute_layout(const std::shared_ptr<const ov::Node>& node) {
    auto concat = ov::as_type_ptr<const ov::op::v0::Concat>(node);
    OPENVINO_ASSERT(concat, "MetalConcatOp expects v0::Concat");
    m_axis = concat->get_axis();
    m_element_type = concat->get_output_element_type(0);

    const auto& out_shape = concat->get_output_shape(0);
    OPENVINO_ASSERT(!out_shape.empty(), "Concat: static output shape required");
    int64_t axis_norm = m_axis;
    if (axis_norm < 0)
        axis_norm += static_cast<int64_t>(out_shape.size());
    OPENVINO_ASSERT(axis_norm >= 0 && axis_norm < static_cast<int64_t>(out_shape.size()),
                    "Concat: axis out of range");

    m_axis_sizes.clear();
    m_axis_offsets.clear();
    m_axis_sizes.reserve(concat->get_input_size());
    m_axis_offsets.reserve(concat->get_input_size());

    m_outer = 1;
    for (size_t i = 0; i < static_cast<size_t>(axis_norm); ++i)
        m_outer *= out_shape[i];

    m_inner = 1;
    for (size_t i = static_cast<size_t>(axis_norm) + 1; i < out_shape.size(); ++i)
        m_inner *= out_shape[i];

    uint64_t offset = 0;
    for (size_t i = 0; i < concat->get_input_size(); ++i) {
        const auto& in_shape = concat->get_input_shape(i);
        OPENVINO_ASSERT(in_shape.size() == out_shape.size(),
                        "Concat: mismatched input rank");
        uint64_t axis_len = static_cast<uint64_t>(in_shape[static_cast<size_t>(axis_norm)]);
        m_axis_sizes.push_back(axis_len);
        m_axis_offsets.push_back(offset);
        offset += axis_len;
    }
}

void MetalConcatOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalConcatOp::execute() {
    OPENVINO_ASSERT(!inputs().empty(), "Concat: no inputs bound");
    MetalTensor& out = require_output();
    if (out.shape.empty()) {
        out.shape = output_shape();
    }
    out.expected_type = m_element_type;
    if (!out.buf.valid()) {
        size_t bytes = ov::shape_size(out.shape) * element_size(m_element_type);
        out.buf = buffer_manager()->allocate(bytes, m_element_type, /*persistent=*/false, /*storageModePrivate=*/true);
    }

    if (!m_queue) {
        m_queue = [m_device newCommandQueue];
    }
    id<MTLCommandBuffer> cb = [m_queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];

    const size_t elem_sz = element_size(m_element_type);
    const size_t rank = out.shape.size();
    int64_t axis_norm = m_axis;
    if (axis_norm < 0)
        axis_norm += static_cast<int64_t>(rank);
    OPENVINO_ASSERT(axis_norm >= 0 && static_cast<size_t>(axis_norm) < rank, "Concat: invalid axis");

    const size_t outer = static_cast<size_t>(m_outer);
    const size_t inner = static_cast<size_t>(m_inner);

    auto cap = [&](const MetalTensor* t) -> size_t {
        return t && t->buf.valid() ? t->buf.size : 0;
    };
    const size_t dst_cap = out.buf.size;

    for (size_t o = 0; o < outer; ++o) {
        for (size_t i = 0; i < inputs().size() && i < m_axis_sizes.size(); ++i) {
            MetalTensor* src = inputs()[i];
            if (!src || !src->buf.valid())
                continue;
            OPENVINO_ASSERT(src->shape.size() == out.shape.size(),
                            "Concat: runtime rank mismatch at input ", i);
            OPENVINO_ASSERT(static_cast<int64_t>(src->shape[static_cast<size_t>(axis_norm)]) ==
                                static_cast<int64_t>(m_axis_sizes[i]),
                            "Concat: runtime axis dim mismatch at input ", i);
            size_t axis_len = static_cast<size_t>(m_axis_sizes[i]);
            size_t src_off = (o * axis_len * inner) * elem_sz;
            size_t dst_off = (o * static_cast<size_t>(out.shape[axis_norm]) * inner +
                              static_cast<size_t>(m_axis_offsets[i]) * inner) *
                             elem_sz;
            size_t bytes = axis_len * inner * elem_sz;
            OPENVINO_ASSERT(src_off + bytes <= cap(src),
                            "Concat: source buffer too small for input ", i,
                            " need=", src_off + bytes, " have=", cap(src));
            OPENVINO_ASSERT(dst_off + bytes <= dst_cap,
                            "Concat: destination overflow for input ", i,
                            " need=", dst_off + bytes, " have=", dst_cap);
            [blit copyFromBuffer:to_mtl(src->buf)
                    sourceOffset:src_off
                        toBuffer:to_mtl(out.buf)
               destinationOffset:dst_off
                        size:bytes];
        }
    }

    [blit endEncoding];
    start_profiling();
    [cb commit];
    [cb waitUntilCompleted];
    stop_profiling_ms();
}

}  // namespace metal_plugin
}  // namespace ov
