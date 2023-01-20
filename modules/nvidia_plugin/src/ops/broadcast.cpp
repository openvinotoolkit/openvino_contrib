// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "broadcast.hpp"

#include <fmt/format.h>

#include <openvino/op/constant.hpp>

#include "converters.hpp"
#include "cuda_operation_registry.hpp"
#include "ngraph/shape.hpp"

namespace ov {
namespace nvidia_gpu {

namespace {
ov::Shape shape_from_constant_value(const ov::Node* constant_node) {
    const ov::op::v0::Constant* constant = dynamic_cast<const ov::op::v0::Constant*>(constant_node);
    Expects(constant);
    return constant->cast_vector<ov::Shape::value_type>();
}
}  // namespace

BroadcastOp::BroadcastOp(const CreationContext& context,
                         const NodeOp& node,
                         IndexCollection&& inputIds,
                         IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    const auto out_shape = node.get_output_shape(0);
    ngraph::Shape in_shape = node.get_input_shape(0);
    Expects(in_shape.size() <= out_shape.size());

    switch (node.get_broadcast_spec().m_type) {
        case ov::op::BroadcastType::NUMPY:
        case ov::op::BroadcastType::BIDIRECTIONAL:
            break;
        case ov::op::BroadcastType::EXPLICIT: {
            const ov::Shape axes_mapping = shape_from_constant_value(node.get_input_node_ptr(2));
            Expects(axes_mapping.size() == in_shape.size());
            ov::Shape in_shape_reshaped(out_shape.size(), 1);
            for (size_t i = 0; i < axes_mapping.size(); ++i) {
                in_shape_reshaped.at(axes_mapping.at(i)) = in_shape.at(i);
            }
            in_shape = in_shape_reshaped;
        } break;
        default:
            throwIEException(fmt::format("Unsupported broadcast mode {}.", node.get_broadcast_spec().m_type));
    }

    broadcast_params_ = NumpyBroadcastParams::create(in_shape, out_shape);
    broadcast_params_->addWorkbufferRequests(immutable_buffer_sizes_);

    const auto element_type = convertDataType<ov::nvidia_gpu::kernel::Type_t>(node.get_input_element_type(0));
    const size_t dst_num_elements = ngraph::shape_size(out_shape);
    const size_t max_threads_per_block = context.device().props().maxThreadsPerBlock;
    kernel_.emplace(element_type, dst_num_elements, max_threads_per_block);
}

void BroadcastOp::Execute(const InferenceRequestContext& context,
                          Inputs inputs,
                          Outputs outputs,
                          const Workbuffers& workbuffers) const {
    const cudaStream_t stream = context.getThreadContext().stream().get();
    (*kernel_)(stream, inputs[0].get(), broadcast_params_->mapper(workbuffers.immutable_buffers), outputs[0].get());
}

WorkbufferRequest BroadcastOp::GetWorkBufferRequest() const { return {immutable_buffer_sizes_, {}}; }

void BroadcastOp::InitSharedImmutableWorkbuffers(const Buffers& buffers) {
    broadcast_params_->initWorkbuffers(buffers);
}

OPERATION_REGISTER(BroadcastOp, Broadcast);

}  // namespace nvidia_gpu
}  // namespace ov
