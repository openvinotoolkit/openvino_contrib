// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "interpolate.hpp"

#include <fmt/format.h>

#include "converters.hpp"
#include "cuda_operation_registry.hpp"
#include "ngraph/shape.hpp"

namespace CUDAPlugin {

namespace {

std::vector<float> getScalesVector(const std::vector<size_t>& input_shape, const std::vector<size_t>& output_shape) {
    size_t num_of_axes = input_shape.size();
    std::vector<float> scales(input_shape.size(), 1.0f);
    for (size_t i = 0; i < num_of_axes; ++i) {
        float scale = output_shape[i] == input_shape[i]
                          ? 1.0f
                          : static_cast<float>(output_shape[i]) / static_cast<float>(input_shape[i]);
        scales[i] = scale;
    }
    return scales;
}

bool isUpscale(const std::vector<float>& scales) {
    bool is_downscale = false;
    bool is_upscale = false;
    for (const auto s : scales) {
        if (s < 1.0) {
            is_downscale = true;
            break;
        } else if (s > 1.0) {
            is_upscale = true;
        }
    }
    return is_upscale && !is_downscale;
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
    if (node.get_attrs().shape_calculation_mode != Interpolate::ShapeCalcMode::sizes) {
        throwIEException(fmt::format(
            "Unsupported calculation mode ({}). Interpolate operation supports only sizes calculation mode({})",
            node.get_attrs().shape_calculation_mode,
            Interpolate::ShapeCalcMode::sizes));
    }
    if (node.get_attrs().nearest_mode != Interpolate::NearestMode::simple) {
        throwIEException(
            fmt::format("Unsupported nearest mode ({}). Interpolate operation supports only simple nearest mode({})",
                        node.get_attrs().nearest_mode,
                        Interpolate::NearestMode::simple));
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
      scales_{getScalesVector(node.get_input_shape(0), node.get_output_shape(0))},
      is_upscale_{isUpscale(scales_)} {
    checkLimitations(node);

    const auto& prop = context.device().props();
    const auto max_threads_per_block = prop.maxThreadsPerBlock;
    const auto strides = is_upscale_ ? in_strides_[0] : out_strides_[0];
    const auto blocks_number = 1 + strides / max_threads_per_block;
    const auto threads_per_block = (blocks_number == 1) ? strides : max_threads_per_block;
    const auto element_type = convertDataType<CUDAPlugin::kernel::Type_t>(node.get_input_element_type(0));

    interpolate_.emplace(blocks_number, threads_per_block, element_type, is_upscale_);
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
    return {{size_in_bytes(in_strides_), size_in_bytes(out_strides_), size_in_bytes(scales_)}, {}};
}

void InterpolateOp::InitSharedImmutableWorkbuffers(const Buffers& buffers) {
    uploadDataToWorkbuffer(buffers[0], in_strides_);
    uploadDataToWorkbuffer(buffers[1], out_strides_);
    uploadDataToWorkbuffer(buffers[2], scales_);
}

OPERATION_REGISTER(InterpolateOp, Interpolate);

}  // namespace CUDAPlugin
