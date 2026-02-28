// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "clamp_cudnn.hpp"

#include <cuda.h>
#include <fmt/format.h>

#include <cmath>

#include <cuda/constant_factory.hpp>
#include <cuda/descriptor_utils.hpp>
#include <cuda/float16.hpp>
#include <cuda/runtime.hpp>
#include <cuda_operation_registry.hpp>
#include <type_traits>

#include "converters.hpp"

namespace ov {
namespace nvidia_gpu {

ClampCuDnnOp::ClampCuDnnOp(const CreationContext& context,
                           const NodeOp& node,
                           IndexCollection&& inputIds,
                           IndexCollection&& outputIds)
    : OperationCuDnn{context, node, move(inputIds), move(outputIds)},
      data_type_{convertDataType<cudnnDataType_t>(node.get_input_element_type(0))},
      op_type_{getCuDnnOpTensorCompType(data_type_, data_type_, data_type_)},
      max_op_desc_{CUDA::DnnOpTensorDescriptor{}.set(CUDNN_OP_TENSOR_MAX, op_type_, CUDNN_PROPAGATE_NAN)},
      min_op_desc_{CUDA::DnnOpTensorDescriptor{}.set(CUDNN_OP_TENSOR_MIN, op_type_, CUDNN_PROPAGATE_NAN)},
      io_desc_{CUDA::makeInputDnnTensorDescr(node, 0)},
      max_min_desc_{CUDA::makeDnnTensorDescr(node.get_input_element_type(0), {1})},
      max_{node.get_max()},
      min_{node.get_min()} {
    OPENVINO_ASSERT(node.get_input_size() == 1, "Node name: ", GetName());
    OPENVINO_ASSERT(node.get_output_size() == 1, "Node name: ", GetName());

    const auto& shape = node.get_input_shape(0);
    OPENVINO_ASSERT(node.get_output_shape(0) == shape, "Node name: ", GetName());

    const auto in_shape_size = node.get_input_shape(0).size();
    if (in_shape_size > max_shape_size) {
        throw_ov_exception(
            fmt::format("ClampCuDnnOp: in_shape_size > max_shape_size: in_shape_size = {}, max_shape_size = {}",
                        in_shape_size,
                        max_shape_size));
    }

    OPENVINO_ASSERT(node.get_output_element_type(0) == node.get_input_element_type(0), "Node name: ", GetName());

    if (min_ > max_) {
        throw_ov_exception(fmt::format("ov::nvidia_gpu::ClampCuDnnOp: Clamp min_ > max_: min_ = {}, max_ = {}", min_, max_));
    }
}

void ClampCuDnnOp::Execute(const InferenceRequestContext& context,
                           Inputs inputTensors,
                           Outputs outputTensors,
                           const Workbuffers& workbuffers) const {
    OPENVINO_ASSERT(inputTensors.size() == 1, "Node name: ", GetName());
    OPENVINO_ASSERT(outputTensors.size() == 1, "Node name: ", GetName());

    const auto& ib = workbuffers.immutable_buffers;
    OPENVINO_ASSERT(ib.size() == 2, "Node name: ", GetName());

    const void* alpha = &CUDA::NumericConst<CUDA::constants::one>(data_type_);
    const void* beta = &CUDA::NumericConst<CUDA::constants::zero>(data_type_);

    // Res = min(max(X[i], Min[0]), Max[0]) ->
    // Temp = max(X[i], Min[0]);
    // Res = min(Temp, Max[0])

    // cudnnOpTensor() works in-place despite the fact it isn't mentioned in cuDNN documentation
    // if for some reason it stops working this way, a temporary workbuffer should be used to store temp result
    context.getThreadContext().dnnHandle().opTensor(max_op_desc_,
                                                    alpha,
                                                    io_desc_,
                                                    inputTensors[0].get(),
                                                    alpha,
                                                    max_min_desc_,
                                                    ib[min_index].get(),
                                                    beta,
                                                    io_desc_,
                                                    outputTensors[0].get());

    context.getThreadContext().dnnHandle().opTensor(min_op_desc_,
                                                    alpha,
                                                    io_desc_,
                                                    outputTensors[0].get(),
                                                    alpha,
                                                    max_min_desc_,
                                                    ib[max_index].get(),
                                                    beta,
                                                    io_desc_,
                                                    outputTensors[0].get());
}

CudaGraphCompatibility ClampCuDnnOp::GetCudaGraphCompatibilityImpl() const { return CudaGraphCompatibility::FULL; }

void ClampCuDnnOp::InitSharedImmutableWorkbuffers(const Buffers& buffers) {
    switch (data_type_) {
        case CUDNN_DATA_FLOAT:
            initBuffers<float>(buffers);
            break;
        case CUDNN_DATA_DOUBLE:
            initBuffers<double>(buffers);
            break;
        case CUDNN_DATA_HALF:
            initBuffers<__half>(buffers);
            break;
        case CUDNN_DATA_INT8:
            initBuffers<int8_t>(buffers);
            break;
#ifdef CUDA_HAS_BF16_TYPE
        case CUDNN_DATA_BFLOAT16:
            initBuffers<__nv_bfloat16>(buffers);
            break;
#endif
        default:
            throw_ov_exception(fmt::format("ClampCuDnnOp: unsupported data_type_ = {}", toString(data_type_)));
    }
}

WorkbufferRequest ClampCuDnnOp::GetWorkBufferRequest() const {
    const auto el_size = elementSize(data_type_);
    return {{el_size, el_size}, {}};
}

namespace {
template <typename T>
T double_to_int(double x, double float_to_int_converter(double)) {
    if (!std::is_integral<T>()) {
        OPENVINO_THROW("Function double_to_int template parameter must be an integral type.");
    }

    x = float_to_int_converter(x);

    double min_t = static_cast<double>(std::numeric_limits<T>::min());
    if (x < min_t) {
        return std::numeric_limits<T>::min();
    }

    double max_t = static_cast<double>(std::numeric_limits<T>::max());
    if (x > max_t) {
        return std::numeric_limits<T>::max();
    }

    return static_cast<T>(x);
}
}

template <typename T>
void ClampCuDnnOp::initBuffers(const Buffers& buffers) const {
    T max{};
    T min{};
    if constexpr (std::is_integral<T>()) {
        max = double_to_int<T>(max_, std::floor);
        min = double_to_int<T>(min_, std::ceil);
    } else {
        max = static_cast<T>(max_);
        min = static_cast<T>(min_);
    }
    const auto el_size = elementSize(data_type_);
    OPENVINO_ASSERT(el_size == sizeof(T), "Node name: ", GetName());

    const auto& stream = CUDA::DefaultStream::stream();
    stream.upload(buffers[max_index], &max, el_size);
    stream.upload(buffers[min_index], &min, el_size);
}

}  // namespace nvidia_gpu
}  // namespace ov
