// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "runtime/metal_op_reshape.hpp"

#include <numeric>
#include <algorithm>

#include "openvino/core/shape_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "kernel_codegen/metal_kernel_compiler.hpp"
#include "mlir/mlir_builder.hpp"
#include "mlir_codegen/codegen_common.hpp"
#include "runtime/metal_logger.hpp"
#include "runtime/metal_op_utils.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

namespace ov {
namespace metal_plugin {

namespace {
inline id<MTLBuffer> to_mtl(const MetalBuffer& buf) {
    return (__bridge id<MTLBuffer>)buf.buffer;
}
}  // namespace

MetalReshapeOp::MetalReshapeOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "Reshape",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_node(node) {
    m_element_type = node->get_output_element_type(0);
    if (node->get_output_partial_shape(0).is_static()) {
        m_target_shape = node->get_output_shape(0);
    }
}

void MetalReshapeOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
    // No kernel needed: reshape is treated as a view; copy avoided.
}

void MetalReshapeOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }
    MetalOp::compile(buffer_manager);
}

void MetalReshapeOp::execute(MetalCommandBufferHandle /*cmd_buf_handle*/) {
    OPENVINO_ASSERT(!inputs().empty(), "Reshape: missing input");
    MetalTensor* src = inputs()[0];
    OPENVINO_ASSERT(src && src->buf.valid(), "Reshape: input buffer is null");
    MetalTensor& dst = require_output();

    ov::Shape out_shape = !m_target_shape.empty() ? m_target_shape : output_shape();
    if (out_shape.empty() && m_node) {
        const ov::Shape in_shape = !src->shape.empty() ? src->shape : m_node->get_input_shape(0);
        if (!in_shape.empty()) {
            if (auto sq = ov::as_type_ptr<const ov::op::v0::Squeeze>(m_node)) {
                std::vector<int64_t> axes;
                if (sq->get_input_size() > 1) {
                    auto axes_const = ov::as_type_ptr<const ov::op::v0::Constant>(
                        sq->input_value(1).get_node_shared_ptr());
                    OPENVINO_ASSERT(axes_const, "Squeeze: axes must be constant");
                    axes = axes_const->cast_vector<int64_t>();
                }
                if (axes.empty()) {
                    for (size_t i = 0; i < in_shape.size(); ++i) {
                        if (in_shape[i] != 1) {
                            out_shape.push_back(in_shape[i]);
                        }
                    }
                } else {
                    auto norm_axes = axes;
                    ov::util::normalize_axes(norm_axes, static_cast<int64_t>(in_shape.size()));
                    for (size_t i = 0; i < in_shape.size(); ++i) {
                        if (std::find(norm_axes.begin(), norm_axes.end(), static_cast<int64_t>(i)) == norm_axes.end()) {
                            out_shape.push_back(in_shape[i]);
                        }
                    }
                }
            } else if (auto uq = ov::as_type_ptr<const ov::op::v0::Unsqueeze>(m_node)) {
                auto axes_const = ov::as_type_ptr<const ov::op::v0::Constant>(
                    uq->input_value(1).get_node_shared_ptr());
                OPENVINO_ASSERT(axes_const, "Unsqueeze: axes must be constant");
                auto axes = axes_const->cast_vector<int64_t>();
                auto norm_axes = axes;
                ov::util::normalize_axes(norm_axes, static_cast<int64_t>(in_shape.size() + axes.size()));
                out_shape = in_shape;
                std::sort(norm_axes.begin(), norm_axes.end());
                for (size_t i = 0; i < norm_axes.size(); ++i) {
                    out_shape.insert(out_shape.begin() + static_cast<std::ptrdiff_t>(norm_axes[i]), 1);
                }
            }
        }
    }
    if (out_shape.empty() && !src->shape.empty()) {
        out_shape = src->shape;  // propagate input shape when output shape is dynamic
    }
    const size_t in_elems = ov::shape_size(src->shape.empty() ? out_shape : src->shape);
    const size_t out_elems = ov::shape_size(out_shape);
    OPENVINO_ASSERT(in_elems == out_elems, "Reshape: element count mismatch");
    const size_t elem_sz = m_element_type.is_dynamic()
                               ? (src->expected_type == ov::element::dynamic ? src->buf.type : src->expected_type).size()
                               : m_element_type.size();
    const size_t need_src = in_elems * elem_sz;
    OPENVINO_ASSERT(src->buf.size >= need_src, "Reshape: input buffer too small");

    dst.shape = out_shape;
    dst.expected_type = m_element_type.is_dynamic() ? src->expected_type : m_element_type;

    // View semantics: reuse underlying buffer, no copy.
    dst.buf = src->buf;

    start_profiling(nullptr);
    stop_profiling_ms(nullptr);
}

MetalTransposeOp::MetalTransposeOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "Transpose",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_node(node),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
    build_desc(node);
}

void MetalTransposeOp::build_desc(const std::shared_ptr<const ov::Node>& node) {
    auto tr = ov::as_type_ptr<const ov::op::v1::Transpose>(node);
    OPENVINO_ASSERT(tr, "MetalTransposeOp expects v1::Transpose");
    m_element_type = tr->get_output_element_type(0);

    if (tr->get_input_partial_shape(0).is_static()) {
        m_desc.in_shape = std::vector<int64_t>(tr->get_input_shape(0).begin(),
                                               tr->get_input_shape(0).end());
    }
    if (tr->get_output_partial_shape(0).is_static()) {
        m_desc.out_shape = std::vector<int64_t>(tr->get_output_shape(0).begin(),
                                                tr->get_output_shape(0).end());
    }

    auto perm_const = ov::as_type_ptr<const ov::op::v0::Constant>(tr->input_value(1).get_node_shared_ptr());
    OPENVINO_ASSERT(perm_const, "Transpose: perm must be constant");
    auto perm = perm_const->cast_vector<int64_t>();
    m_desc.perm = perm;
}

void MetalTransposeOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalTransposeOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }
    MetalKernelCompiler compiler(m_device ? m_device : (id<MTLDevice>)buffer_manager->device());
    std::string log;
    TransposeCodegenDesc desc{};
    desc.element_type = m_element_type == ov::element::dynamic ? ov::element::f32 : m_element_type;
    desc.use_half = desc.element_type == ov::element::f16;
    desc.use_int = desc.element_type == ov::element::i32;
    desc.in_shape.reserve(m_desc.in_shape.size());
    desc.out_shape.reserve(m_desc.out_shape.size());
    desc.perm.reserve(m_desc.perm.size());
    for (auto v : m_desc.in_shape) desc.in_shape.push_back(static_cast<uint32_t>(v));
    for (auto v : m_desc.out_shape) desc.out_shape.push_back(static_cast<uint32_t>(v));
    for (auto v : m_desc.perm) desc.perm.push_back(static_cast<uint32_t>(v));
    mlir::MLIRContext ctx;
    auto module = build_mlir_transpose_from_model(make_single_op_model(m_node), ctx);
    auto source = generate_msl_from_mlir(module, desc);
    m_pipeline = compiler.compile_msl_from_source(source, "transpose_kernel", log);
    OPENVINO_ASSERT(m_pipeline, "MetalTransposeOp: failed to compile kernel: ", log);

    MetalOp::compile(buffer_manager);
}

void MetalTransposeOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(!inputs().empty(), "Transpose: missing input");
    MetalTensor* src = inputs()[0];
    OPENVINO_ASSERT(src && src->buf.valid(), "Transpose: input buffer null");
    MetalTensor& dst = require_output();

    ov::Shape in_shape = !src->shape.empty()
                             ? src->shape
                             : ov::Shape(m_desc.in_shape.begin(), m_desc.in_shape.end());
    ov::Shape out_shape = !dst.shape.empty()
                              ? dst.shape
                              : ov::Shape(m_desc.out_shape.begin(), m_desc.out_shape.end());
    OPENVINO_ASSERT(!in_shape.empty() && !out_shape.empty(), "Transpose: shapes unknown");

    if (!dst.buf.valid()) {
        size_t bytes = ov::shape_size(out_shape) * m_element_type.size();
        dst.buf = buffer_manager()->allocate(bytes, m_element_type, /*persistent=*/false, dst.prefer_private);
    }
    dst.shape = out_shape;
    dst.expected_type = m_element_type;

    const uint32_t num_elems = static_cast<uint32_t>(ov::shape_size(out_shape));

    id<MTLCommandBuffer> cb = static_cast<id<MTLCommandBuffer>>(cmd_buf_handle);
    OPENVINO_ASSERT(cb, "MetalTransposeOp: command buffer is null");
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m_pipeline];
    [enc setBuffer:to_mtl(src->buf) offset:0 atIndex:0];
    [enc setBuffer:to_mtl(dst.buf) offset:0 atIndex:1];
    const size_t elem_sz = m_element_type.size();
    const size_t need_src = ov::shape_size(in_shape) * elem_sz;
    const size_t need_dst = ov::shape_size(out_shape) * elem_sz;
    OPENVINO_ASSERT(src->buf.size >= need_src, "Transpose: src buffer too small");
    OPENVINO_ASSERT(dst.buf.size >= need_dst, "Transpose: dst buffer too small");
    [enc setBytes:&num_elems length:sizeof(num_elems) atIndex:2];

    MTLSize grid = MTLSizeMake(num_elems, 1, 1);
    const NSUInteger tg_size = static_cast<NSUInteger>(metal_clamp_tg_size((void*)m_pipeline, 64));
    MTLSize tg = MTLSizeMake(tg_size, 1, 1);

    start_profiling(enc);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    stop_profiling_ms(enc);
    [enc endEncoding];
}

}  // namespace metal_plugin
}  // namespace ov
