// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "split_kernel_ir.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/core/validation_util.hpp"
#include "slice_kernel_ir.hpp"
#include "slice_kernel_ir.hpp"

namespace ov {
namespace metal_plugin {

namespace {

std::shared_ptr<const ov::Node> find_split_node(const std::shared_ptr<const ov::Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::v1::Split>(node.get()) || ov::is_type<ov::op::v1::VariadicSplit>(node.get())) {
            return node;
        }
    }
    return {};
}

std::vector<size_t> extract_split_sizes(const std::shared_ptr<const ov::Node>& node, int64_t& axis_out) {
    if (auto split = ov::as_type_ptr<const ov::op::v1::Split>(node)) {
        auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(split->input_value(1).get_node_shared_ptr());
        OPENVINO_ASSERT(axis_const, "Split axis must be constant");
        auto axis_vec = axis_const->cast_vector<int64_t>();
        OPENVINO_ASSERT(!axis_vec.empty(), "Split axis empty");
        axis_out = axis_vec[0];
        const size_t parts = split->get_num_splits();
        const auto rank = split->get_input_partial_shape(0).rank().get_length();
        int64_t axis_norm = axis_out >= 0 ? axis_out : axis_out + static_cast<int64_t>(rank);
        OPENVINO_ASSERT(axis_norm >= 0 && axis_norm < static_cast<int64_t>(rank), "Split axis out of range");
        auto dim = split->get_input_shape(0).at(static_cast<size_t>(axis_norm));
        OPENVINO_ASSERT(dim % parts == 0, "Split: dimension not divisible by number of splits");
        size_t chunk = dim / parts;
        return std::vector<size_t>(parts, chunk);
    } else if (auto vs = ov::as_type_ptr<const ov::op::v1::VariadicSplit>(node)) {
        auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(vs->input_value(1).get_node_shared_ptr());
        OPENVINO_ASSERT(axis_const, "VariadicSplit axis must be constant");
        auto axis_vec = axis_const->cast_vector<int64_t>();
        OPENVINO_ASSERT(!axis_vec.empty(), "VariadicSplit axis empty");
        axis_out = axis_vec[0];
        auto lengths_const = ov::as_type_ptr<const ov::op::v0::Constant>(vs->get_input_node_shared_ptr(2));
        OPENVINO_ASSERT(lengths_const, "VariadicSplit: split lengths must be constant for Metal backend");
        auto lengths = lengths_const->cast_vector<int64_t>();
        std::vector<size_t> res;
        res.reserve(lengths.size());
        for (auto v : lengths) {
            OPENVINO_ASSERT(v >= 0, "VariadicSplit: negative lengths are not supported");
            res.push_back(static_cast<size_t>(v));
        }
        return res;
    }
    OPENVINO_THROW("Unsupported Split node type");
}

}  // namespace

MetalKernelIR build_kernel_ir_for_split(const std::shared_ptr<const ov::Model>& model) {
    MetalKernelIR ir;
    auto node = find_split_node(model);
    OPENVINO_ASSERT(node, "Split builder: no Split/VariadicSplit node found in model");

    const auto in_shape = node->get_input_shape(0);
    const auto dtype = resolve_metal_dtype(node->get_input_element_type(0));
    KernelTensor in0{"in0", {in_shape.begin(), in_shape.end()}};
    in0.dtype = dtype;
    ir.tensors.push_back(in0);

    int64_t axis = 0;
    auto split_sizes = extract_split_sizes(node, axis);
    std::vector<int64_t> prefix;
    prefix.reserve(split_sizes.size());
    int64_t acc = 0;
    for (auto s : split_sizes) {
        prefix.push_back(acc);
        acc += static_cast<int64_t>(s);
    }

    int64_t axis_norm = axis >= 0 ? axis : axis + static_cast<int64_t>(in_shape.size());
    for (size_t i = 0; i < split_sizes.size(); ++i) {
        auto out_shape = in_shape;
        out_shape[static_cast<size_t>(axis_norm)] = split_sizes[i];
        KernelTensor out{"out" + std::to_string(i), {out_shape.begin(), out_shape.end()}};
        out.dtype = resolve_metal_dtype(node->get_output_element_type(i));
        ir.tensors.push_back(out);
    }

    // Expand Split into multiple Slice ops (one per output).
    size_t rank = in_shape.size();
    auto axis_norm_runtime = axis >= 0 ? axis : axis + static_cast<int64_t>(rank);
    int64_t offset = 0;
    for (size_t idx = 0; idx < split_sizes.size(); ++idx) {
        const auto out_idx = idx + 1;
        const auto out_shape = ir.tensors[out_idx].shape;
        std::vector<int64_t> starts(rank, 0);
        starts[axis_norm_runtime] = offset;
        std::vector<int64_t> steps(rank, 1);

        auto op = make_slice_op({in_shape.begin(), in_shape.end()},
                                starts,
                                steps,
                                out_shape,
                                node->get_output_element_type(0),
                                ir.tensors[0],
                                ir.tensors[out_idx]);
        ir.ops.push_back(op);
        offset += static_cast<int64_t>(split_sizes[idx]);
    }

    return ir;
}

}  // namespace metal_plugin
}  // namespace ov
