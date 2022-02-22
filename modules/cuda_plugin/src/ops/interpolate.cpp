// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "interpolate.hpp"

#include <fmt/format.h>

#include "converters.hpp"
#include "cuda_operation_registry.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/validation_util.hpp"

namespace CUDAPlugin {

namespace {

std::vector<float> getScalesVector(const CUDAPlugin::InterpolateOp::NodeOp& node) {
    // for calculation scale for nearest mode see
    // https://docs.openvino.ai/2021.1/openvino_docs_ops_image_Interpolate_4.html
    Expects(node.get_attrs().mode == ngraph::op::v4::Interpolate::InterpolateMode::nearest);

    const auto scales = ngraph::get_constant_from_source(node.input_value(2))->cast_vector<float>();
    const auto axis = ngraph::get_constant_from_source(node.input_value(3))->cast_vector<int64_t>();

    const auto& input_shape = node.get_input_shape(0);
    const auto& output_shape = node.get_output_shape(0);

    std::vector<float> result_scales(node.get_output_shape(0).size(), 1.0f);
    for (size_t i = 0; i < axis.size(); ++i) {
        using ShapeCalcMode = ngraph::op::v4::Interpolate::ShapeCalcMode;
        const auto idx = axis[i];
        if (node.get_attrs().shape_calculation_mode == ShapeCalcMode::scales) {
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

bool canApplyUpscaleOptimizing(const std::vector<float>& scales) {
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

void checkLimitations(const InterpolateOp::NodeOp& node) {
    using namespace ngraph::op::v4;
    if (node.get_input_shape(0).size() != 4u) {
        throwIEException(fmt::format("Unsupported shape rank {}. Interpolate operation supports only 4d tensor",
                                     node.get_input_shape(0).size()));
    }
    if (node.get_attrs().mode != Interpolate::InterpolateMode::nearest) {
        throwIEException(fmt::format("Unsupported mode ({}). Interpolate operation supports only neares mode({})",
                                     node.get_attrs().mode,
                                     Interpolate::InterpolateMode::nearest));
    }
    if (node.get_attrs().nearest_mode != Interpolate::NearestMode::simple &&
        node.get_attrs().nearest_mode != Interpolate::NearestMode::floor) {
        throwIEException(
            fmt::format("Unsupported nearest mode ({}). Interpolate operation supports only simple ({}), floor({}) and "
                        "round_prefer_floor({}) modes",
                        node.get_attrs().nearest_mode,
                        Interpolate::NearestMode::simple,
                        Interpolate::NearestMode::floor));
    }
    if (node.get_attrs().coordinate_transformation_mode != Interpolate::CoordinateTransformMode::asymmetric &&
        node.get_attrs().coordinate_transformation_mode != Interpolate::CoordinateTransformMode::tf_half_pixel_for_nn) {
        throwIEException(
            fmt::format("Unsupported coordinate transfrom mode ({}). Interpolate operation asymmetric ({}) and "
                        "tf_half_pixel_for_nn ({}) modes",
                        node.get_attrs().coordinate_transformation_mode,
                        Interpolate::CoordinateTransformMode::asymmetric,
                        Interpolate::CoordinateTransformMode::tf_half_pixel_for_nn));
    }
    if (node.get_attrs().antialias != false) {
        throwIEException(
            fmt::format("Unsupported antialias mode ({}). Interpolate operation supports only antialias set({})",
                        node.get_attrs().antialias,
                        false));
    }
    if (!std::all_of(
            node.get_attrs().pads_begin.cbegin(), node.get_attrs().pads_begin.cend(), [](int i) { return i == 0; })) {
        throwIEException(fmt::format("Unsupported begin pads. Interpolate operation supports all pads are equal 0"));
    }
    if (!std::all_of(
            node.get_attrs().pads_end.cbegin(), node.get_attrs().pads_end.cend(), [](int i) { return i == 0; })) {
        throwIEException(fmt::format("Unsupported end pads. Interpolate operation supports all pads are equal 0"));
    }
}

}  // namespace

InterpolateOp::InterpolateOp(const CreationContext& context,
                             const NodeOp& node,
                             IndexCollection&& inputIds,
                             IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)),
      in_strides_{ngraph::row_major_strides(node.get_input_shape(0))},
      out_strides_{ngraph::row_major_strides(node.get_output_shape(0))},
      scales_{getScalesVector(node)},
      in_shape_{node.get_input_shape(0)},
      out_shape_{node.get_output_shape(0)},
      can_use_upscale_optimizing_{canApplyUpscaleOptimizing(scales_)} {
    checkLimitations(node);

    const auto& prop = context.device().props();
    const auto max_threads_per_block = prop.maxThreadsPerBlock;
    // is_upscale_ = false;
    const auto strides = can_use_upscale_optimizing_ ? in_shape_[0] * in_strides_[0] : out_shape_[0] * out_strides_[0];
    const auto blocks_number = 1 + strides / max_threads_per_block;
    const auto threads_per_block = (blocks_number == 1) ? strides : max_threads_per_block;
    const auto element_type = convertDataType<CUDAPlugin::kernel::Type_t>(node.get_input_element_type(0));

    interpolate_ = kernel::Interpolate(
        blocks_number,
        threads_per_block,
        element_type,
        can_use_upscale_optimizing_,
        static_cast<kernel::Interpolate::NearestMode>(node.get_attrs().nearest_mode),
        static_cast<kernel::Interpolate::TransformMode>(node.get_attrs().coordinate_transformation_mode));
}

void InterpolateOp::Execute(const InferenceRequestContext& context,
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

template <typename T>
static auto size_in_bytes(const std::vector<T>& v) noexcept {
    return sizeof(T) * v.size();
}

template <typename T>
static void uploadDataToWorkbuffer(CUDA::DevicePointer<void*> buffer, const std::vector<T>& v) {
    auto& stream = CUDA::DefaultStream::stream();
    stream.upload(buffer, v.data(), size_in_bytes(v));
}

WorkbufferRequest InterpolateOp::GetWorkBufferRequest() const {
    return {{size_in_bytes(in_strides_), size_in_bytes(out_strides_), size_in_bytes(scales_),size_in_bytes(in_shape_), size_in_bytes(out_shape_)}, {}};
}

void InterpolateOp::InitSharedImmutableWorkbuffers(const Buffers& buffers) {
    uploadDataToWorkbuffer(buffers[0], in_strides_);
    uploadDataToWorkbuffer(buffers[1], out_strides_);
    uploadDataToWorkbuffer(buffers[2], scales_);
    uploadDataToWorkbuffer(buffers[3], in_shape_);
    uploadDataToWorkbuffer(buffers[4], out_shape_);
}

OPERATION_REGISTER(InterpolateOp, Interpolate);

}  // namespace CUDAPlugin
