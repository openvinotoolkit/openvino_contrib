// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <array>
#include <cuda_runtime.h>
#include <gsl/gsl_assert>
#include <utility>
#include "details/cuda_ngraph_import.hpp"
#include <cuda_operation_registry.hpp>
#include <cuda/device.hpp>
#include <gpu/device_pointers.hpp>

#include <cuda/cuda_type_traits.hpp>
#include "convert.hpp"

namespace CUDAPlugin {

namespace kernel {

template<typename TOutput, typename TInput>
__global__ void convert_impl(size_t inputSize, TOutput * out, const TInput *in) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < inputSize) {
    if constexpr(std::is_same_v<TInput, __half> || std::is_same_v<TInput, __nv_bfloat16> ||
                 std::is_same_v<TOutput, __half> || std::is_same_v<TOutput, __nv_bfloat16>) {
      // workaround for "error: more than one conversion function from "const __half" to "..." applies"
      // converting __half, __nv_bfloat16 via float
      out[i] = static_cast<TOutput>(static_cast<float>(in[i]));
    } else {
      out[i] = static_cast<TOutput>(in[i]);
    }
  }
}
} //namespace kernel

ConvertOp::ConvertOp(const CUDA::Device& device,
                     const std::shared_ptr<ngraph::Node>& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds)
    : OperationBase(device, node, std::move(inputIds), std::move(outputIds)) {
    Type_t input_element_type = node->get_input_element_type(0);
    Type_t output_element_type = node->get_output_element_type(0);
    Expects(input_element_type >= Type_t::boolean && input_element_type <= Type_t::u64);
    Expects(output_element_type >= Type_t::boolean && output_element_type <= Type_t::u64);
    if (input_element_type == Type_t::u1 || output_element_type == Type_t::u1)
      THROW_IE_EXCEPTION << "Unsupported data type : " << Type_t::u1;
    auto input_shape = node->get_input_shape(0);
    auto output_shape = node->get_output_shape(0);
    size_ = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());
    auto output_size_ = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<size_t>());
    Expects(size_ == output_size_);
    convert_kernel_ = getConvertKernel(output_element_type, input_element_type);
}

void ConvertOp::Execute(const InferenceRequestContext& context, Inputs inputs, Outputs outputs, const Workbuffers&) {
    Expects(inputs.size() == 1);
    Expects(outputs.size() == 1);
    const auto& stream = context.getThreadContext().stream();
    const unsigned maxBlockSize = CudaDevice::GetMaxGridBlockSizeParams(context.getThreadContext().device().currentId());
    const unsigned numBlocks = (size_ % maxBlockSize == 0) ?
                               (size_ / maxBlockSize) :
                               (size_ / maxBlockSize + 1);
    const unsigned threadsPerBlock = (numBlocks == 1) ? size_ : maxBlockSize;
    convert_kernel_(stream, size_, outputs[0], inputs[0], numBlocks, threadsPerBlock);
}

OPERATION_REGISTER(ConvertOp, Convert);

namespace detail {
using namespace kernel;

template<size_t OutputType, size_t InputType>
struct Convert {
  static void function(const CUDA::Stream& stream, size_t size,
                       InferenceEngine::gpu::DevicePointer<void*> output,
                       InferenceEngine::gpu::DevicePointer<const void*> input,
                       unsigned numBlocks, unsigned threadsPerBlock) {
    using namespace InferenceEngine::gpu;
    using namespace ngraph;
    using namespace ngraph::element;
    constexpr Type_t output_type =  static_cast<Type_t>(OutputType + static_cast<size_t>(Type_t::boolean));
    constexpr Type_t input_type =  static_cast<Type_t>(InputType + static_cast<size_t>(Type_t::boolean));
    using TOutput = typename cuda_type_traits<output_type>::value_type;
    using TInput = typename cuda_type_traits<input_type>::value_type;
    if (OutputType == InputType) {
      if (output.get() == input.get()) return;
      throwIfError(cudaMemcpyAsync(output.get(), input.get(),
                                   size * sizeof(TOutput),
                                   cudaMemcpyDeviceToDevice, stream.get()));
    } else {
      convert_impl<TOutput, TInput><<<numBlocks, threadsPerBlock, 0, stream.get()>>>(
          size, static_cast<TOutput *>(output.get()), static_cast<const TInput *>(input.get()));
    }
  }
};

using Type_t = ngraph::element::Type_t;
using convert_t = ConvertOp::convert_t;

constexpr size_t type_count = static_cast<size_t>(Type_t::u64) - static_cast<size_t>(Type_t::boolean) + 1;

template<template<size_t> class Template>
struct convert_vector : std::array<convert_t, type_count> {
  constexpr convert_vector() : convert_vector(std::make_index_sequence<type_count>()) {}
private:
  template<size_t ... I>
  constexpr convert_vector(std::index_sequence<I...>) : std::array<convert_t, type_count> { &Template<I>::function ... } {}
};

template<template<size_t, size_t> class Template, size_t N>
struct reduce {
  template<size_t M>
  using type = Template<N, M>;
};

template<template<size_t, size_t> class Template>
class convert_matrix : public std::array<std::array<convert_t, type_count>, type_count> {
public:
  constexpr convert_matrix() : convert_matrix<Template>(std::make_index_sequence<type_count>()) {}
private:
  template<size_t ... I>
  constexpr convert_matrix(std::index_sequence<I...>) : std::array<std::array<convert_t, type_count>, type_count> {
    convert_vector<reduce<Template, I>::template type>{} ... } {}
};
} //namespace detail

ConvertOp::convert_t ConvertOp::getConvertKernel(Type_t output_element_type, Type_t input_element_type) {
  static constexpr detail::convert_matrix<detail::Convert> matrix {};
  const size_t input_type_index = static_cast<size_t>(input_element_type) - static_cast<size_t>(Type_t::boolean);
  const size_t output_type_index = static_cast<size_t>(output_element_type) - static_cast<size_t>(Type_t::boolean);
  return matrix[output_type_index][input_type_index];
}
} // namespace CUDAPlugin

