// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "runtime/metal_op_split.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/core/validation_util.hpp"
#include "runtime/metal_logger.hpp"

namespace ov {
namespace metal_plugin {

namespace {
inline id<MTLBuffer> to_mtl(const MetalBuffer& buf) {
    return (__bridge id<MTLBuffer>)buf.buffer;
}

std::vector<size_t> extract_split_sizes(const std::shared_ptr<const ov::Node>& node,
                                        int64_t& axis_out,
                                        ov::Shape& input_shape) {
    if (auto s = ov::as_type_ptr<const ov::op::v1::Split>(node)) {
        auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(s->input_value(1).get_node_shared_ptr());
        OPENVINO_ASSERT(axis_const, "Split axis must be constant");
        axis_out = axis_const->cast_vector<int64_t>().at(0);
        input_shape = s->get_input_shape(0);
        const size_t parts = s->get_num_splits();
        const size_t axis_norm = axis_out >= 0 ? static_cast<size_t>(axis_out)
                                               : static_cast<size_t>(axis_out + static_cast<int64_t>(input_shape.size()));
        OPENVINO_ASSERT(axis_norm < input_shape.size(), "Split axis out of range");
        OPENVINO_ASSERT(input_shape.at(axis_norm) % parts == 0, "Split dimension not divisible by parts");
        size_t chunk = input_shape.at(axis_norm) / parts;
        return std::vector<size_t>(parts, chunk);
    } else if (auto vs = ov::as_type_ptr<const ov::op::v1::VariadicSplit>(node)) {
        auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(vs->input_value(1).get_node_shared_ptr());
        OPENVINO_ASSERT(axis_const, "VariadicSplit axis must be constant");
        axis_out = axis_const->cast_vector<int64_t>().at(0);
        input_shape = vs->get_input_shape(0);
        auto lengths_const = ov::as_type_ptr<const ov::op::v0::Constant>(vs->input_value(2).get_node_shared_ptr());
        OPENVINO_ASSERT(lengths_const, "VariadicSplit lengths must be constant");
        auto lengths = lengths_const->cast_vector<int64_t>();
        std::vector<size_t> res;
        res.reserve(lengths.size());
        for (auto v : lengths) {
            OPENVINO_ASSERT(v >= 0, "VariadicSplit negative length not supported");
            res.push_back(static_cast<size_t>(v));
        }
        return res;
    }
    OPENVINO_THROW("MetalSplitOp: unsupported node type");
}

}  // namespace

MetalSplitOp::MetalSplitOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "Split",
              {},  // output shapes are per-output; not used in base
              device,
              queue),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
    parse_split(node);
}

void MetalSplitOp::parse_split(const std::shared_ptr<const ov::Node>& node) {
    m_split_sizes = extract_split_sizes(node, m_axis, m_input_shape);
    m_element_type = node->get_input_element_type(0);
}

void MetalSplitOp::set_outputs(const std::vector<std::unique_ptr<MetalTensor>>& outputs) {
    m_outputs.clear();
    m_outputs.reserve(outputs.size());
    for (const auto& o : outputs) {
        m_outputs.push_back(o.get());
    }
}

void MetalSplitOp::execute() {
    OPENVINO_ASSERT(!inputs().empty(), "Split: no inputs bound");
    MetalTensor* src = inputs().at(0);
    OPENVINO_ASSERT(src && src->buf.valid(), "Split: input buffer is null");
    // Some graphs (e.g. num_splits = 1) route through set_output(), so fill m_outputs lazily.
    if (m_outputs.empty() && output()) {
        m_outputs.push_back(output());
    }
    OPENVINO_ASSERT(!m_outputs.empty(), "Split: outputs not bound");

    const ov::Shape shape = !src->shape.empty() ? src->shape : m_input_shape;
    OPENVINO_ASSERT(!shape.empty(), "Split: input shape unknown");
    int64_t axis_norm = m_axis;
    if (axis_norm < 0)
        axis_norm += static_cast<int64_t>(shape.size());
    OPENVINO_ASSERT(axis_norm >= 0 && axis_norm < static_cast<int64_t>(shape.size()), "Split: axis out of range");

    size_t axis_len = shape[static_cast<size_t>(axis_norm)];
    size_t outer = 1;
    for (size_t i = 0; i < static_cast<size_t>(axis_norm); ++i)
        outer *= shape[i];
    size_t inner = 1;
    for (size_t i = static_cast<size_t>(axis_norm) + 1; i < shape.size(); ++i)
        inner *= shape[i];

    // If split sizes were not known statically (e.g., zeros), derive from output shapes if present.
    size_t total_requested = 0;
    for (auto s : m_split_sizes) total_requested += s;
    if (total_requested == 0 || total_requested != axis_len) {
        m_split_sizes.clear();
        m_split_sizes.reserve(m_outputs.size());
        for (auto* out : m_outputs) {
            size_t sz = out && out->shape.size() > static_cast<size_t>(axis_norm)
                            ? out->shape[static_cast<size_t>(axis_norm)]
                            : 0;
            m_split_sizes.push_back(sz);
        }
    }
    size_t sum = 0;
    for (auto s : m_split_sizes) sum += s;
    OPENVINO_ASSERT(sum == axis_len, "Split: split sizes do not sum to axis length");

    // Allocate outputs and set shapes/types.
    for (size_t i = 0; i < m_outputs.size(); ++i) {
        MetalTensor* out = m_outputs[i];
        size_t split = m_split_sizes[i];
        if (out) {
            ov::Shape out_shape = shape;
            out_shape[static_cast<size_t>(axis_norm)] = split;
            out->shape = out_shape;
            out->expected_type = m_element_type;
            size_t bytes = element_size() * outer * split * inner;
            if (!out->buf.valid()) {
                out->buf = buffer_manager()->allocate(bytes,
                                                      m_element_type,
                                                      /*persistent=*/false,
                                                      /*storageModePrivate=*/true);
            }
        }
    }

    if (!m_queue) {
        m_queue = [m_device newCommandQueue];
    }
    id<MTLCommandBuffer> cb = [m_queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];

    size_t axis_offset = 0;
    for (size_t out_idx = 0; out_idx < m_outputs.size(); ++out_idx) {
        MetalTensor* out = m_outputs[out_idx];
        if (!out || !out->buf.valid()) {
            axis_offset += m_split_sizes[out_idx];
            continue;
        }
        const size_t split = m_split_sizes[out_idx];
        for (size_t o = 0; o < outer; ++o) {
            size_t src_off = (o * axis_len * inner + axis_offset * inner) * element_size();
            size_t dst_off = (o * split * inner) * element_size();
            size_t bytes = split * inner * element_size();
            [blit copyFromBuffer:to_mtl(src->buf)
                    sourceOffset:src_off
                        toBuffer:to_mtl(out->buf)
               destinationOffset:dst_off
                        size:bytes];
        }
        axis_offset += split;
    }
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

}  // namespace metal_plugin
}  // namespace ov
