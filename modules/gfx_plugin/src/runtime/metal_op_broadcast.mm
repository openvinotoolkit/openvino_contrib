// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "runtime/metal_op_broadcast.hpp"

#include <numeric>

#include "openvino/core/except.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "runtime/metal_logger.hpp"
#include "runtime/metal_op_utils.hpp"
#include "mlir_builder.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir_codegen/codegen_common.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
inline id<MTLBuffer> to_mtl(const MetalBuffer& buf) {
    return (__bridge id<MTLBuffer>)buf.buffer;
}

std::vector<int> make_strides(const ov::Shape& shp) {
    if (shp.empty())
        return {1};
    std::vector<int> st(shp.size(), 1);
    for (int i = static_cast<int>(shp.size()) - 2; i >= 0; --i) {
        st[i] = st[i + 1] * static_cast<int>(shp[i + 1]);
    }
    return st;
}

std::vector<int64_t> get_constant_i64(const ov::Output<ov::Node>& input) {
    auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(input.get_node_shared_ptr());
    if (!c)
        return {};
    return c->cast_vector<int64_t>();
}

}  // namespace

MetalBroadcastOp::MetalBroadcastOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "Broadcast",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_node(node),
      m_element_type(node->get_output_element_type(0)),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {}

void MetalBroadcastOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalBroadcastOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }

    MetalKernelCompiler compiler(m_device ? m_device : (id<MTLDevice>)buffer_manager->device());
    std::string log;
    mlir::MLIRContext ctx;
    auto module = build_mlir_broadcast_from_model(make_single_op_model(m_node), ctx);
    BroadcastCodegenDesc desc{};
    desc.element_type = m_element_type;
    auto source = generate_msl_from_mlir(module, desc);
    m_pipeline = compiler.compile_msl_from_source(source, "broadcast_kernel", log);
    OPENVINO_ASSERT(m_pipeline, "MetalBroadcastOp: failed to compile broadcast kernel: ", log);

    ov::Shape in_shape = m_node->get_input_shape(0);
    ov::Shape out_shape = output_shape();
    if (in_shape.empty())
        in_shape = ov::Shape{1};
    if (out_shape.empty())
        out_shape = ov::Shape{1};

    m_num_elems = static_cast<uint32_t>(ov::shape_size(out_shape));
    m_in_dims.assign(in_shape.begin(), in_shape.end());
    m_out_dims.assign(out_shape.begin(), out_shape.end());
    if (m_in_dims.empty())
        m_in_dims.push_back(1);
    if (m_out_dims.empty())
        m_out_dims.push_back(1);
    m_in_strides = make_strides(in_shape);
    if (m_in_strides.empty())
        m_in_strides.push_back(1);

    const size_t in_rank = m_in_dims.size();
    const size_t out_rank = m_out_dims.size();
    OPENVINO_ASSERT(in_rank <= 8 && out_rank <= 8,
                    "Broadcast: rank > 8 not supported by kernel");
    m_axes.assign(in_rank, 0);

    // Validate/compute axes mapping.
    if (auto b3 = std::dynamic_pointer_cast<const ov::op::v3::Broadcast>(m_node)) {
        auto spec = b3->get_broadcast_spec();
        if (spec.m_type == ov::op::BroadcastType::NONE || spec.m_type == ov::op::BroadcastType::EXPLICIT) {
            auto axes = get_constant_i64(m_node->input_value(2));
            OPENVINO_ASSERT(!axes.empty(), "Broadcast explicit: axes_mapping must be constant");
            OPENVINO_ASSERT(axes.size() == in_rank, "Broadcast explicit: axes size mismatch");
            for (size_t i = 0; i < axes.size(); ++i) {
                int64_t axis = axes[i];
                OPENVINO_ASSERT(axis >= 0 && axis < static_cast<int64_t>(out_rank),
                                "Broadcast explicit: axis out of range");
                m_axes[i] = static_cast<int>(axis);
            }
        } else {
            int64_t start_axis = (spec.m_type == ov::op::BroadcastType::PDPD && spec.m_axis != -1)
                                     ? spec.m_axis
                                     : static_cast<int64_t>(out_rank) - static_cast<int64_t>(in_rank);
            OPENVINO_ASSERT(start_axis >= 0, "Broadcast: invalid start_axis");
            OPENVINO_ASSERT(start_axis + static_cast<int64_t>(in_rank) <= static_cast<int64_t>(out_rank),
                            "Broadcast: start_axis out of range");
            for (size_t i = 0; i < in_rank; ++i) {
                m_axes[i] = static_cast<int>(start_axis + static_cast<int64_t>(i));
            }
        }
    } else if (auto b1 = std::dynamic_pointer_cast<const ov::op::v1::Broadcast>(m_node)) {
        auto spec = b1->get_broadcast_spec();
        if (spec.m_type == ov::op::AutoBroadcastType::NONE) {
            auto axes = get_constant_i64(m_node->input_value(2));
            OPENVINO_ASSERT(!axes.empty(), "Broadcast explicit: axes_mapping must be constant");
            OPENVINO_ASSERT(axes.size() == in_rank, "Broadcast explicit: axes size mismatch");
            for (size_t i = 0; i < axes.size(); ++i) {
                int64_t axis = axes[i];
                OPENVINO_ASSERT(axis >= 0 && axis < static_cast<int64_t>(out_rank),
                                "Broadcast explicit: axis out of range");
                m_axes[i] = static_cast<int>(axis);
            }
        } else {
            int64_t start_axis = (spec.m_type == ov::op::AutoBroadcastType::PDPD && spec.m_axis != -1)
                                     ? spec.m_axis
                                     : static_cast<int64_t>(out_rank) - static_cast<int64_t>(in_rank);
            OPENVINO_ASSERT(start_axis >= 0, "Broadcast: invalid start_axis");
            OPENVINO_ASSERT(start_axis + static_cast<int64_t>(in_rank) <= static_cast<int64_t>(out_rank),
                            "Broadcast: start_axis out of range");
            for (size_t i = 0; i < in_rank; ++i) {
                m_axes[i] = static_cast<int>(start_axis + static_cast<int64_t>(i));
            }
        }
    } else {
        OPENVINO_THROW("MetalBroadcastOp: unsupported broadcast version");
    }

    // Validate broadcast compatibility.
    for (size_t i = 0; i < in_rank; ++i) {
        const int axis = m_axes[i];
        OPENVINO_ASSERT(axis >= 0 && axis < static_cast<int>(out_rank), "Broadcast: axis out of range");
        const int in_dim = m_in_dims[i];
        const int out_dim = m_out_dims[static_cast<size_t>(axis)];
        OPENVINO_ASSERT(in_dim == 1 || in_dim == out_dim, "Broadcast: incompatible dimensions");
    }

    MetalOp::compile(buffer_manager);
}

void MetalBroadcastOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(inputs().size() >= 1, "Broadcast: missing input");
    MetalTensor* src = inputs()[0];
    OPENVINO_ASSERT(src && src->buf.valid(), "Broadcast: input buffer null");

    MetalTensor& dst = require_output();
    ov::Shape in_shape = !src->shape.empty() ? src->shape : ov::Shape{};
    if (in_shape.empty() && m_node->get_input_partial_shape(0).is_static()) {
        in_shape = m_node->get_input_shape(0);
    }
    ov::Shape out_shape = !dst.shape.empty() ? dst.shape : output_shape();
    if (out_shape.empty() && m_node->get_output_partial_shape(0).is_static()) {
        out_shape = m_node->get_output_shape(0);
    }
    OPENVINO_ASSERT(!in_shape.empty() && !out_shape.empty(), "Broadcast: runtime shape unknown");

    const size_t in_bytes = m_element_type.size() * ov::shape_size(in_shape);
    OPENVINO_ASSERT(src->buf.size >= in_bytes, "Broadcast: input buffer too small");

    const size_t in_rank = in_shape.size();
    const size_t out_rank = out_shape.size();
    OPENVINO_ASSERT(in_rank <= 8 && out_rank <= 8,
                    "Broadcast: rank > 8 not supported by kernel");

    m_num_elems = static_cast<uint32_t>(ov::shape_size(out_shape));
    m_in_dims.assign(in_shape.begin(), in_shape.end());
    m_out_dims.assign(out_shape.begin(), out_shape.end());
    if (m_in_dims.empty())
        m_in_dims.push_back(1);
    if (m_out_dims.empty())
        m_out_dims.push_back(1);
    m_in_strides = make_strides(in_shape);
    if (m_in_strides.empty())
        m_in_strides.push_back(1);

    m_axes.assign(in_rank, 0);
    if (auto b3 = std::dynamic_pointer_cast<const ov::op::v3::Broadcast>(m_node)) {
        auto spec = b3->get_broadcast_spec();
        if (spec.m_type == ov::op::BroadcastType::NONE || spec.m_type == ov::op::BroadcastType::EXPLICIT) {
            auto axes = get_constant_i64(m_node->input_value(2));
            OPENVINO_ASSERT(!axes.empty(), "Broadcast explicit: axes_mapping must be constant");
            OPENVINO_ASSERT(axes.size() == in_rank, "Broadcast explicit: axes size mismatch");
            for (size_t i = 0; i < axes.size(); ++i) {
                int64_t axis = axes[i];
                OPENVINO_ASSERT(axis >= 0 && axis < static_cast<int64_t>(out_rank),
                                "Broadcast explicit: axis out of range");
                m_axes[i] = static_cast<int>(axis);
            }
        } else {
            int64_t start_axis = (spec.m_type == ov::op::BroadcastType::PDPD && spec.m_axis != -1)
                                     ? spec.m_axis
                                     : static_cast<int64_t>(out_rank) - static_cast<int64_t>(in_rank);
            OPENVINO_ASSERT(start_axis >= 0, "Broadcast: invalid start_axis");
            OPENVINO_ASSERT(start_axis + static_cast<int64_t>(in_rank) <= static_cast<int64_t>(out_rank),
                            "Broadcast: start_axis out of range");
            for (size_t i = 0; i < in_rank; ++i) {
                m_axes[i] = static_cast<int>(start_axis + static_cast<int64_t>(i));
            }
        }
    } else if (auto b1 = std::dynamic_pointer_cast<const ov::op::v1::Broadcast>(m_node)) {
        auto spec = b1->get_broadcast_spec();
        if (spec.m_type == ov::op::AutoBroadcastType::NONE) {
            auto axes = get_constant_i64(m_node->input_value(2));
            OPENVINO_ASSERT(!axes.empty(), "Broadcast explicit: axes_mapping must be constant");
            OPENVINO_ASSERT(axes.size() == in_rank, "Broadcast explicit: axes size mismatch");
            for (size_t i = 0; i < axes.size(); ++i) {
                int64_t axis = axes[i];
                OPENVINO_ASSERT(axis >= 0 && axis < static_cast<int64_t>(out_rank),
                                "Broadcast explicit: axis out of range");
                m_axes[i] = static_cast<int>(axis);
            }
        } else {
            int64_t start_axis = (spec.m_type == ov::op::AutoBroadcastType::PDPD && spec.m_axis != -1)
                                     ? spec.m_axis
                                     : static_cast<int64_t>(out_rank) - static_cast<int64_t>(in_rank);
            OPENVINO_ASSERT(start_axis >= 0, "Broadcast: invalid start_axis");
            OPENVINO_ASSERT(start_axis + static_cast<int64_t>(in_rank) <= static_cast<int64_t>(out_rank),
                            "Broadcast: start_axis out of range");
            for (size_t i = 0; i < in_rank; ++i) {
                m_axes[i] = static_cast<int>(start_axis + static_cast<int64_t>(i));
            }
        }
    } else {
        OPENVINO_THROW("MetalBroadcastOp: unsupported broadcast version");
    }

    const size_t bytes = m_element_type.size() * static_cast<size_t>(m_num_elems);
    if (!dst.buf.valid() || dst.buf.size < bytes) {
        dst.buf = buffer_manager()->allocate(bytes, m_element_type, /*persistent=*/false, dst.prefer_private);
    }
    dst.expected_type = m_element_type;
    dst.shape = out_shape;

    id<MTLCommandBuffer> cb = static_cast<id<MTLCommandBuffer>>(cmd_buf_handle);
    OPENVINO_ASSERT(cb, "MetalBroadcastOp: command buffer is null");
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m_pipeline];

    [enc setBuffer:to_mtl(src->buf) offset:0 atIndex:0];
    [enc setBuffer:to_mtl(dst.buf) offset:0 atIndex:1];
    uint32_t num = m_num_elems;
    uint32_t out_rank_u = static_cast<uint32_t>(m_out_dims.size());
    uint32_t in_rank_u = static_cast<uint32_t>(m_in_dims.size());
    [enc setBytes:&num length:sizeof(num) atIndex:2];
    [enc setBytes:&out_rank_u length:sizeof(out_rank_u) atIndex:3];
    [enc setBytes:&in_rank_u length:sizeof(in_rank_u) atIndex:4];
    [enc setBytes:m_out_dims.data() length:m_out_dims.size() * sizeof(int) atIndex:5];
    [enc setBytes:m_in_dims.data() length:m_in_dims.size() * sizeof(int) atIndex:6];
    [enc setBytes:m_in_strides.data() length:m_in_strides.size() * sizeof(int) atIndex:7];
    [enc setBytes:m_axes.data() length:m_axes.size() * sizeof(int) atIndex:8];

    const NSUInteger threads = 64;
    MTLSize grid = MTLSizeMake(num, 1, 1);
    const NSUInteger tg_size = static_cast<NSUInteger>(metal_clamp_tg_size((void*)m_pipeline, threads));
    MTLSize tg = MTLSizeMake(tg_size, 1, 1);
    start_profiling(enc);
    if (num == 0) {
        [enc endEncoding];
        return;
    }
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    stop_profiling_ms(enc);
    [enc endEncoding];
    dst.expected_type = m_element_type;
}

}  // namespace gfx_plugin
}  // namespace ov
