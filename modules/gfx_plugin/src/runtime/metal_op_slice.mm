// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "runtime/metal_op_slice.hpp"

#include <algorithm>
#include <numeric>

#include "openvino/core/shape_util.hpp"
#include "openvino/op/constant.hpp"
#include "kernel_codegen/metal_kernel_compiler.hpp"
#include "mlir/mlir_builder.hpp"
#include "runtime/metal_logger.hpp"
#include "runtime/metal_op_utils.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir_codegen/codegen_common.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
inline id<MTLBuffer> to_mtl(const MetalBuffer& buf) {
    return (__bridge id<MTLBuffer>)buf.buffer;
}

std::vector<int64_t> get_const_i64(const std::shared_ptr<const ov::Node>& n) {
    auto c = ov::as_type_ptr<const ov::op::v0::Constant>(n);
    OPENVINO_ASSERT(c, "Slice: inputs must be Constant");
    return c->cast_vector<int64_t>();
}

std::vector<int64_t> normalize_axes(const std::vector<int64_t>& axes, size_t rank) {
    std::vector<int64_t> out;
    out.reserve(axes.size());
    for (auto a : axes) {
        int64_t ax = a;
        if (ax < 0)
            ax += static_cast<int64_t>(rank);
        OPENVINO_ASSERT(ax >= 0 && ax < static_cast<int64_t>(rank), "Slice: axis out of range");
        out.push_back(ax);
    }
    return out;
}
}  // namespace

MetalSliceOp::MetalSliceOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "Slice",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_node(node),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
    parse_slice(node);
}

void MetalSliceOp::parse_slice(const std::shared_ptr<const ov::Node>& node) {
    auto sl = ov::as_type_ptr<const ov::op::v8::Slice>(node);
    OPENVINO_ASSERT(sl, "MetalSliceOp expects v8::Slice");

    const auto in_shape = node->get_input_shape(0);
    OPENVINO_ASSERT(!in_shape.empty(), "Slice: input shape must be static");
    const size_t rank = in_shape.size();

    m_element_type = node->get_output_element_type(0);
    OPENVINO_ASSERT(m_element_type == ov::element::f16 || m_element_type == ov::element::f32 ||
                        m_element_type == ov::element::i32 || m_element_type == ov::element::i64,
                    "Slice: element type not supported");

    auto starts = get_const_i64(node->get_input_node_shared_ptr(1));
    auto ends = get_const_i64(node->get_input_node_shared_ptr(2));
    auto steps = get_const_i64(node->get_input_node_shared_ptr(3));

    std::vector<int64_t> axes;
    if (node->get_input_size() > 4) {
        axes = get_const_i64(node->get_input_node_shared_ptr(4));
    } else {
        axes.resize(starts.size());
        std::iota(axes.begin(), axes.end(), 0);
    }
    OPENVINO_ASSERT(starts.size() == ends.size() && starts.size() == steps.size() &&
                        starts.size() == axes.size(),
                    "Slice: starts/ends/steps/axes size mismatch");

    axes = normalize_axes(axes, rank);

    m_starts.assign(rank, 0);
    m_steps.assign(rank, 1);
    m_out_shape.assign(rank, 0);

    for (size_t i = 0; i < rank; ++i)
        m_out_shape[i] = static_cast<uint32_t>(in_shape[i]);

    for (size_t i = 0; i < axes.size(); ++i) {
        const size_t axis = static_cast<size_t>(axes[i]);
        int64_t dim = static_cast<int64_t>(in_shape[axis]);
        int64_t step = steps[i];
        OPENVINO_ASSERT(step > 0, "Slice: only positive steps supported");
        int64_t start = starts[i];
        int64_t end = ends[i];
        if (start < 0) start += dim;
        if (end < 0) end += dim;
        start = std::max<int64_t>(0, std::min<int64_t>(start, dim));
        end = std::max<int64_t>(0, std::min<int64_t>(end, dim));
        if (end < start) end = start;
        int64_t len = (end - start + step - 1) / step;
        m_starts[axis] = static_cast<int32_t>(start);
        m_steps[axis] = static_cast<uint32_t>(step);
        m_out_shape[axis] = static_cast<uint32_t>(len);
    }

    // Precompute input strides (row-major).
    m_in_stride.assign(rank, 1);
    for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
        m_in_stride[static_cast<size_t>(i)] =
            m_in_stride[static_cast<size_t>(i + 1)] * static_cast<uint32_t>(in_shape[static_cast<size_t>(i + 1)]);
    }
}

void MetalSliceOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalSliceOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }
    MetalKernelCompiler compiler(m_device ? m_device : (id<MTLDevice>)buffer_manager->device());
    std::string log;
    ConvertCodegenDesc desc{};
    desc.dst_type = m_element_type;
    desc.element_type = m_element_type == ov::element::dynamic ? ov::element::f32 : m_element_type;
    mlir::MLIRContext ctx;
    auto module = build_mlir_slice_from_model(make_single_op_model(m_node), ctx);
    auto source = generate_msl_for_slice_generic(desc, module);
    m_pipeline = compiler.compile_msl_from_source(source, "slice_kernel", log);
    OPENVINO_ASSERT(m_pipeline, "MetalSliceOp: failed to compile kernel: ", log);

    MetalOp::compile(buffer_manager);
}

void MetalSliceOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(!inputs().empty(), "Slice: missing input");
    MetalTensor* src = inputs()[0];
    OPENVINO_ASSERT(src && src->buf.valid(), "Slice: input buffer null");
    MetalTensor& dst = require_output();

    ov::Shape in_shape = !src->shape.empty() ? src->shape : ov::Shape{};
    if (in_shape.empty() && m_node->get_input_partial_shape(0).is_static()) {
        in_shape = m_node->get_input_shape(0);
    }
    OPENVINO_ASSERT(!in_shape.empty(), "Slice: input shape unknown");
    if (m_node->get_input_partial_shape(0).is_static()) {
        OPENVINO_ASSERT(in_shape == m_node->get_input_shape(0),
                        "Slice: runtime input shape mismatch");
    }

    ov::Shape out_shape;
    out_shape.reserve(m_out_shape.size());
    for (auto v : m_out_shape) out_shape.push_back(static_cast<size_t>(v));

    const size_t bytes = ov::shape_size(out_shape) * m_element_type.size();
    if (!dst.buf.valid() || dst.buf.size < bytes) {
        dst.buf = buffer_manager()->allocate(bytes, m_element_type, /*persistent=*/false, dst.prefer_private);
    }
    dst.shape = out_shape;
    dst.expected_type = m_element_type;

    const uint32_t rank = static_cast<uint32_t>(m_out_shape.size());
    const uint32_t total = static_cast<uint32_t>(ov::shape_size(out_shape));
    const size_t in_bytes = ov::shape_size(in_shape) * m_element_type.size();
    OPENVINO_ASSERT(src->buf.size >= in_bytes, "Slice: input buffer too small");
    if (total == 0) {
        return;
    }

    id<MTLCommandBuffer> cb = static_cast<id<MTLCommandBuffer>>(cmd_buf_handle);
    OPENVINO_ASSERT(cb, "MetalSliceOp: command buffer is null");
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m_pipeline];
    [enc setBuffer:to_mtl(src->buf) offset:0 atIndex:0];
    [enc setBuffer:to_mtl(dst.buf) offset:0 atIndex:1];
    [enc setBytes:&total length:sizeof(total) atIndex:2];
    [enc setBytes:&rank length:sizeof(rank) atIndex:3];
    [enc setBytes:m_out_shape.data() length:m_out_shape.size() * sizeof(uint32_t) atIndex:4];
    [enc setBytes:m_in_stride.data() length:m_in_stride.size() * sizeof(uint32_t) atIndex:5];
    [enc setBytes:m_starts.data() length:m_starts.size() * sizeof(int32_t) atIndex:6];
    [enc setBytes:m_steps.data() length:m_steps.size() * sizeof(uint32_t) atIndex:7];

    MTLSize grid = MTLSizeMake(total, 1, 1);
    const NSUInteger tg_size = static_cast<NSUInteger>(metal_clamp_tg_size((void*)m_pipeline, 256));
    MTLSize tg = MTLSizeMake(tg_size, 1, 1);

    start_profiling(enc);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    stop_profiling_ms(enc);
    [enc endEncoding];
}

}  // namespace gfx_plugin
}  // namespace ov
