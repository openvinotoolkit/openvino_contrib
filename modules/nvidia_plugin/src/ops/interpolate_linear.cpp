// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "interpolate_linear.hpp"

#include <fmt/format.h>

#include "interpolate_components/interpolate_components.hpp"
#include "ops/converters.hpp"

namespace ov {
namespace nvidia_gpu {

namespace {

void checkLimitations(const InterpolateLinearOp::NodeOp& node) {
    if (node.get_input_shape(0).size() > kernel::InterpolateLinear::MAX_SHAPE_RANK) {
        throw_ov_exception(
            fmt::format("Unsupported shape rank {}. InterpolateLinearOp operation supports up to {} dimensions.",
                        node.get_input_shape(0).size(),
                        kernel::InterpolateLinear::MAX_SHAPE_RANK));
    }
    if (!std::all_of(
            node.get_attrs().pads_begin.cbegin(), node.get_attrs().pads_begin.cend(), [](int i) { return i == 0; })) {
        throw_ov_exception(
            fmt::format("Unsupported begin pads. InterpolateLinearOp operation supports all pads equal to 0."));
    }
    if (!std::all_of(
            node.get_attrs().pads_end.cbegin(), node.get_attrs().pads_end.cend(), [](int i) { return i == 0; })) {
        throw_ov_exception(
            fmt::format("Unsupported end pads. InterpolateLinearOp operation supports all pads equal to 0."));
    }
}

}  // namespace

InterpolateLinearOp::InterpolateLinearOp(const CreationContext& context,
                                         const NodeOp& node,
                                         IndexCollection&& inputIds,
                                         IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    OPENVINO_ASSERT(
        node.get_attrs().mode == ov::op::v4::Interpolate::InterpolateMode::LINEAR, "Node name: ", GetName());
    checkLimitations(node);

    std::vector<size_t> axes;
    std::vector<float> scales;
    Interpolate::Details::getAxesAndScales(node, axes, scales);

    const auto transform_mode = static_cast<kernel::InterpolateLinear::CoordinateTransformMode>(
        node.get_attrs().coordinate_transformation_mode);
    const auto element_type = convertDataType<ov::nvidia_gpu::kernel::Type_t>(node.get_input_element_type(0));
    const auto max_threads_per_block = context.device().props().maxThreadsPerBlock;

    interpolate_ = kernel::InterpolateLinear(node.get_input_shape(0),
                                             axes,
                                             scales,
                                             node.get_output_shape(0),
                                             transform_mode,
                                             node.get_attrs().antialias,
                                             element_type,
                                             max_threads_per_block);
}

void InterpolateLinearOp::Execute(const InferenceRequestContext& context,
                                  Inputs inputs,
                                  Outputs outputs,
                                  const Workbuffers& workbuffers) const {
    (*interpolate_)(context.getThreadContext().stream().get(), inputs[0].get(), outputs[0].get());
}

CudaGraphCompatibility InterpolateLinearOp::GetCudaGraphCompatibility() const { return CudaGraphCompatibility::FULL; }

WorkbufferRequest InterpolateLinearOp::GetWorkBufferRequest() const {
    return {interpolate_->immutableWorkbufferSizes(), {}};
}

void InterpolateLinearOp::InitSharedImmutableWorkbuffers(const Buffers& in_buffers) {
    std::vector<void*> buffers;
    std::transform(in_buffers.begin(), in_buffers.end(), std::back_inserter(buffers), [](auto v) { return v.get(); });
    interpolate_->initImmutableWorkbuffers(buffers);
}

}  // namespace nvidia_gpu
}  // namespace ov
