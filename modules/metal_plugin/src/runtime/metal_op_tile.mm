// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "runtime/metal_op_tile.hpp"

#include <limits>

#include "openvino/core/except.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/tile.hpp"
#include "runtime/metal_logger.hpp"
#include "runtime/metal_op_utils.hpp"
#include "mlir_builder.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir_codegen/codegen_common.hpp"

namespace ov {
namespace metal_plugin {

namespace {
inline id<MTLBuffer> to_mtl(const MetalBuffer& buf) {
    return (__bridge id<MTLBuffer>)buf.buffer;
}

std::vector<int> make_strides(const ov::Shape& shp) {
    if (shp.empty()) return {1};
    std::vector<int> st(shp.size(), 1);
    for (int i = static_cast<int>(shp.size()) - 2; i >= 0; --i) {
        st[i] = st[i + 1] * static_cast<int>(shp[i + 1]);
    }
    return st;
}
}  // namespace

MetalTileOp::MetalTileOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "Tile",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_node(node),
      m_element_type(node->get_output_element_type(0)),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {}

void MetalTileOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalTileOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }

    MetalKernelCompiler compiler(m_device ? m_device : (id<MTLDevice>)buffer_manager->device());
    std::string log;
    mlir::MLIRContext ctx;
    auto module = build_mlir_tile_from_model(make_single_op_model(m_node), ctx);
    TileCodegenDesc desc{};
    desc.element_type = m_element_type;
    auto source = generate_msl_from_mlir(module, desc);
    m_pipeline = compiler.compile_msl_from_source(source, "tile_kernel", log);
    OPENVINO_ASSERT(m_pipeline, "MetalTileOp: failed to compile tile kernel: ", log);

    auto in_shape = m_node->get_input_shape(0);
    auto out_shape = output_shape();
    m_num_elems = static_cast<uint32_t>(ov::shape_size(out_shape));

    m_in_dims.assign(in_shape.begin(), in_shape.end());
    if (m_in_dims.empty()) m_in_dims.push_back(1);
    m_out_dims.assign(out_shape.begin(), out_shape.end());
    if (m_out_dims.empty()) m_out_dims.push_back(1);

    m_in_strides = make_strides(in_shape);
    if (m_in_strides.empty()) m_in_strides.push_back(1);
    m_out_strides = make_strides(out_shape);
    if (m_out_strides.empty()) m_out_strides.push_back(1);

    MetalOp::compile(buffer_manager);
}

void MetalTileOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(inputs().size() >= 1, "Tile: missing input");
    MetalTensor* src = inputs()[0];
    OPENVINO_ASSERT(src && src->buf.valid(), "Tile: input buffer null");

    MetalTensor& dst = require_output();
    ov::Shape in_shape = !src->shape.empty() ? src->shape : ov::Shape{};
    if (in_shape.empty() && m_node->get_input_partial_shape(0).is_static()) {
        in_shape = m_node->get_input_shape(0);
    }
    OPENVINO_ASSERT(!in_shape.empty(), "Tile: input shape unknown");
    ov::Shape out_shape = !dst.shape.empty() ? dst.shape : output_shape();
    if (out_shape.empty() && m_node->get_output_partial_shape(0).is_static()) {
        out_shape = m_node->get_output_shape(0);
    }
    OPENVINO_ASSERT(!out_shape.empty(), "Tile: output shape unknown");

    m_num_elems = static_cast<uint32_t>(ov::shape_size(out_shape));
    m_in_dims.assign(in_shape.begin(), in_shape.end());
    if (m_in_dims.empty()) m_in_dims.push_back(1);
    m_out_dims.assign(out_shape.begin(), out_shape.end());
    if (m_out_dims.empty()) m_out_dims.push_back(1);
    m_in_strides = make_strides(in_shape);
    if (m_in_strides.empty()) m_in_strides.push_back(1);
    m_out_strides = make_strides(out_shape);
    if (m_out_strides.empty()) m_out_strides.push_back(1);

    const size_t in_bytes = m_element_type.size() * ov::shape_size(in_shape);
    OPENVINO_ASSERT(src->buf.size >= in_bytes, "Tile: input buffer too small");
    const size_t bytes = m_element_type.size() * static_cast<size_t>(m_num_elems);
    if (!dst.buf.valid() || dst.buf.size < bytes) {
        dst.buf = buffer_manager()->allocate(bytes, m_element_type, /*persistent=*/false, dst.prefer_private);
    }
    dst.expected_type = m_element_type;
    dst.shape = out_shape;

    id<MTLCommandBuffer> cb = static_cast<id<MTLCommandBuffer>>(cmd_buf_handle);
    OPENVINO_ASSERT(cb, "MetalTileOp: command buffer is null");
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m_pipeline];

    [enc setBuffer:to_mtl(src->buf) offset:0 atIndex:0];
    [enc setBuffer:to_mtl(dst.buf) offset:0 atIndex:1];
    uint32_t num = m_num_elems;
    uint32_t rank = static_cast<uint32_t>(m_out_dims.size());
    [enc setBytes:&num length:sizeof(num) atIndex:2];
    [enc setBytes:&rank length:sizeof(rank) atIndex:3];
    [enc setBytes:m_out_dims.data() length:m_out_dims.size() * sizeof(int) atIndex:4];
    [enc setBytes:m_in_dims.data() length:m_in_dims.size() * sizeof(int) atIndex:5];
    [enc setBytes:m_out_strides.data() length:m_out_strides.size() * sizeof(int) atIndex:6];
    [enc setBytes:m_in_strides.data() length:m_in_strides.size() * sizeof(int) atIndex:7];

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

    dst.expected_type = m_element_type;
}

}  // namespace metal_plugin
}  // namespace ov
