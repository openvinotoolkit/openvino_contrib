// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "broadcast.hpp"

#include <fmt/format.h>

#include <ngraph/op/constant.hpp>

#include "converters.hpp"
#include "cuda_operation_registry.hpp"
#include "ngraph/shape.hpp"

namespace CUDAPlugin {

namespace {
ngraph::Shape shape_from_constant_value(const ngraph::Node* constant_node) {
    const ngraph::op::v0::Constant* constant = dynamic_cast<const ngraph::op::v0::Constant*>(constant_node);
    Expects(constant);
    return constant->cast_vector<ngraph::Shape::value_type>();
}
}  // namespace

BroadcastOp::BroadcastOp(const CreationContext& context,
                         const NodeOp& node,
                         IndexCollection&& inputIds,
                         IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    auto out_shape = node.get_output_shape(0);
    const size_t shape_rank = out_shape.size();

    ngraph::Shape in_shape = node.get_input_shape(0);
    Expects(in_shape.size() <= shape_rank);

    switch (node.get_broadcast_spec().m_type) {
        case ngraph::op::BroadcastType::NUMPY:
        case ngraph::op::BroadcastType::BIDIRECTIONAL:
            while (in_shape.size() < shape_rank) {
                in_shape.insert(in_shape.begin(), 1);
            }
            break;
        case ngraph::op::BroadcastType::EXPLICIT: {
            const ngraph::Shape axes_mapping = shape_from_constant_value(node.get_input_node_ptr(2));
            Expects(axes_mapping.size() == in_shape.size());
            ngraph::Shape in_shape_reshaped(out_shape.size(), 1);
            for (size_t i = 0; i < axes_mapping.size(); ++i) {
                in_shape_reshaped.at(axes_mapping.at(i)) = in_shape.at(i);
            }
            in_shape = in_shape_reshaped;
        } break;
        default:
            throwIEException(fmt::format("Unsupported broadcast mode {}.", node.get_broadcast_spec().m_type));
    }
    Expects(in_shape.size() == shape_rank);

    for (size_t i = 0; i < shape_rank; ++i) {
        auto in_dim = in_shape.at(i);
        broadcast_dims_.emplace_back(in_dim == 1 ? 0 : 1);
        Expects((in_dim == 1) || (in_dim == out_shape.at(i)));
    }
    Expects(broadcast_dims_.size() == shape_rank);

    src_strides_ = ngraph::row_major_strides(in_shape);
    dst_strides_ = ngraph::row_major_strides(out_shape);

    const auto element_type = convertDataType<CUDAPlugin::kernel::Type_t>(node.get_input_element_type(0));
    const size_t dst_num_elements = ngraph::shape_size(out_shape);

    const size_t max_threads_per_block = context.device().props().maxThreadsPerBlock;
    kernel_.emplace(element_type, shape_rank, dst_num_elements, max_threads_per_block);
}

void BroadcastOp::Execute(const InferenceRequestContext& context,
                          Inputs inputs,
                          Outputs outputs,
                          const Workbuffers& workbuffers) const {
    const cudaStream_t stream = context.getThreadContext().stream().get();
    const void* src = inputs[0].get();
    void* dst = outputs[0].get();
    const size_t* broadcast_dims = static_cast<const size_t*>(workbuffers.immutable_buffers[0].get());
    const size_t* src_strides = static_cast<const size_t*>(workbuffers.immutable_buffers[1].get());
    const size_t* dst_strides = static_cast<const size_t*>(workbuffers.immutable_buffers[2].get());
    (*kernel_)(stream, src, dst, broadcast_dims, src_strides, dst_strides);
}

template <typename T>
static auto size_in_bytes(const std::vector<T>& v) noexcept {
    return sizeof(T) * v.size();
}

template <typename T>
static void uploadDataToWorkbuffer(CUDA::DevicePointer<void*> buffer, const std::vector<T>& v) {
    auto& stream = CUDA::DefaultStream::stream();
    stream.upload(buffer, v.data(), size_in_bytes(v));
}

WorkbufferRequest BroadcastOp::GetWorkBufferRequest() const {
    return {{size_in_bytes(broadcast_dims_), size_in_bytes(src_strides_), size_in_bytes(dst_strides_)}, {}};
}

void BroadcastOp::InitSharedImmutableWorkbuffers(const Buffers& buffers) {
    uploadDataToWorkbuffer(buffers[0], broadcast_dims_);
    uploadDataToWorkbuffer(buffers[1], src_strides_);
    uploadDataToWorkbuffer(buffers[2], dst_strides_);
}

OPERATION_REGISTER(BroadcastOp, Broadcast);

}  // namespace CUDAPlugin
