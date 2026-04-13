// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "interpolate_nearest.hpp"

#include <fmt/format.h>

#include "openvino/op/constant.hpp"
#include "cuda_operation_registry.hpp"
#include "ops/converters.hpp"

namespace ov {
namespace nvidia_gpu {

namespace {

std::vector<float> getScalesVector(const ov::nvidia_gpu::InterpolateNearestOp::NodeOp& node) {
    // for calculation scale for nearest mode see
    // https://docs.openvino.ai/2021.1/openvino_docs_ops_image_Interpolate_4.html
    const auto scales_const = ov::as_type_ptr<op::v0::Constant>(node.input_value(2).get_node_shared_ptr());
    OPENVINO_ASSERT(scales_const);
    const auto scales = scales_const->cast_vector<float>();
    std::vector<int64_t> axis;
    if (node.inputs().size() > 3) {
        const auto axis_const = ov::as_type_ptr<op::v0::Constant>(node.input_value(3).get_node_shared_ptr());
        OPENVINO_ASSERT(axis_const);
        axis = axis_const->cast_vector<int64_t>();
    } else {
        axis.resize(node.get_input_partial_shape(0).rank().get_length());
        std::iota(axis.begin(), axis.end(), 0);
    }

    const auto& input_shape = node.get_input_shape(0);
    const auto& output_shape = node.get_output_shape(0);

    std::vector<float> result_scales(node.get_output_shape(0).size(), 1.0f);
    for (size_t i = 0; i < axis.size(); ++i) {
        using ShapeCalcMode = ov::op::v4::Interpolate::ShapeCalcMode;
        const auto idx = axis[i];
        if (node.get_attrs().shape_calculation_mode == ShapeCalcMode::SCALES) {
            result_scales[idx] = scales[i];
        } else {
            float scale = output_shape[idx] == input_shape[idx]
                              ? 1.0f
                              : static_cast<float>(output_shape[idx]) / static_cast<float>(input_shape[idx]);
            result_scales[idx] = scale;
        }
    }
    return result_scales;
}

bool canApplyUpscaleOptimizing(const InterpolateNearestOp::NodeOp& node, const std::vector<float>& scales) {
    using CoordinateTransformMode = ov::op::v4::Interpolate::CoordinateTransformMode;
    switch (node.get_attrs().coordinate_transformation_mode) {
        case CoordinateTransformMode::ASYMMETRIC:
        case CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN:
            break;
        default:
            return false;
    };

    using NearestMode = ov::op::v4::Interpolate::NearestMode;
    switch (node.get_attrs().nearest_mode) {
        case NearestMode::SIMPLE:
        case NearestMode::FLOOR:
            break;
        default:
            return false;
    };

    bool is_downscale = false;
    bool can_be_optimized = false;
    for (const auto s : scales) {
        if (s < 1.0) {
            is_downscale = true;
            break;
        } else if (s > 1.0 && s - std::floor(s) == 0.f) {
            can_be_optimized = true;
        }
    }
    return can_be_optimized && !is_downscale;
}

void checkLimitations(const InterpolateNearestOp::NodeOp& node) {
    using namespace ov::op::v4;
    if (node.get_input_shape(0).size() != 4u) {
        throw_ov_exception(
            fmt::format("Unsupported shape rank {}. InterpolateNearestOp operation supports only 4d tensor",
                        node.get_input_shape(0).size()));
    }
    if (node.get_attrs().antialias != false) {
        throw_ov_exception(fmt::format(
            "Unsupported antialias mode ({}). InterpolateNearestOp operation supports only antialias set({})",
            node.get_attrs().antialias,
            false));
    }
    if (!std::all_of(
            node.get_attrs().pads_begin.cbegin(), node.get_attrs().pads_begin.cend(), [](int i) { return i == 0; })) {
        throw_ov_exception(
            fmt::format("Unsupported begin pads. InterpolateNearestOp operation supports all pads are equal 0"));
    }
    if (!std::all_of(
            node.get_attrs().pads_end.cbegin(), node.get_attrs().pads_end.cend(), [](int i) { return i == 0; })) {
        throw_ov_exception(
            fmt::format("Unsupported end pads. InterpolateNearestOp operation supports all pads are equal 0"));
    }
}

}  // namespace

InterpolateNearestOp::InterpolateNearestOp(const CreationContext& context,
                                           const NodeOp& node,
                                           IndexCollection&& inputIds,
                                           IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)),
      in_strides_{ov::row_major_strides(node.get_input_shape(0))},
      out_strides_{ov::row_major_strides(node.get_output_shape(0))},
      scales_{getScalesVector(node)},
      in_shape_{node.get_input_shape(0)},
      out_shape_{node.get_output_shape(0)},
      can_use_upscale_optimizing_{canApplyUpscaleOptimizing(node, scales_)} {
    OPENVINO_ASSERT(
        node.get_attrs().mode == ov::op::v4::Interpolate::InterpolateMode::NEAREST, "Node name: ", GetName());
    checkLimitations(node);

    const auto& prop = context.device().props();
    const auto max_threads_per_block = prop.maxThreadsPerBlock;

    const auto strides = can_use_upscale_optimizing_ ? in_shape_[0] * in_strides_[0] : out_shape_[0] * out_strides_[0];
    const auto blocks_number = 1 + strides / max_threads_per_block;
    const auto threads_per_block = (blocks_number == 1) ? strides : max_threads_per_block;
    const auto element_type = convertDataType<ov::nvidia_gpu::kernel::Type_t>(node.get_input_element_type(0));

    interpolate_ =
        kernel::InterpolateNearest(blocks_number,
                                   threads_per_block,
                                   element_type,
                                   can_use_upscale_optimizing_,
                                   static_cast<kernel::InterpolateNearest::NearestMode>(node.get_attrs().nearest_mode),
                                   static_cast<kernel::InterpolateNearest::CoordinateTransformMode>(
                                       node.get_attrs().coordinate_transformation_mode));
}

void InterpolateNearestOp::Execute(const InferenceRequestContext& context,
                                   Inputs inputs,
                                   Outputs outputs,
                                   const Workbuffers& workbuffers) const {
    const cudaStream_t stream = context.getThreadContext().stream().get();
    const void* src = inputs[0].get();
    void* dst = outputs[0].get();
    (*interpolate_)(stream,
                    src,
                    static_cast<const size_t*>(workbuffers.immutable_buffers[0].get()),
                    static_cast<const size_t*>(workbuffers.immutable_buffers[1].get()),
                    static_cast<const float*>(workbuffers.immutable_buffers[2].get()),
                    static_cast<const size_t*>(workbuffers.immutable_buffers[3].get()),
                    static_cast<const size_t*>(workbuffers.immutable_buffers[4].get()),
                    dst);
}

CudaGraphCompatibility InterpolateNearestOp::GetCudaGraphCompatibilityImpl() const { return CudaGraphCompatibility::FULL; }

template <typename T>
static auto size_in_bytes(const std::vector<T>& v) noexcept {
    return sizeof(T) * v.size();
}

template <typename T>
static void uploadDataToWorkbuffer(CUDA::DevicePointer<void*> buffer, const std::vector<T>& v) {
    auto& stream = CUDA::DefaultStream::stream();
    stream.upload(buffer, v.data(), size_in_bytes(v));
}

WorkbufferRequest InterpolateNearestOp::GetWorkBufferRequest() const {
    return {{size_in_bytes(in_strides_),
             size_in_bytes(out_strides_),
             size_in_bytes(scales_),
             size_in_bytes(in_shape_),
             size_in_bytes(out_shape_)},
            {}};
}

void InterpolateNearestOp::InitSharedImmutableWorkbuffers(const Buffers& buffers) {
    uploadDataToWorkbuffer(buffers[0], in_strides_);
    uploadDataToWorkbuffer(buffers[1], out_strides_);
    uploadDataToWorkbuffer(buffers[2], scales_);
    uploadDataToWorkbuffer(buffers[3], in_shape_);
    uploadDataToWorkbuffer(buffers[4], out_shape_);
}

}  // namespace nvidia_gpu
}  // namespace ov
