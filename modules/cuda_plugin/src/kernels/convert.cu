// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <array>
#include "convert.cuh"
#include "convert.hpp"
#include "error.hpp"

namespace CUDAPlugin {
namespace kernel {

template <typename TOutput, typename TInput>
__global__
    void convert_impl(size_t inputSize, TOutput* out, const TInput* in) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < inputSize) {
        out[i] = cast<TOutput>(in[i]);
    }
}

namespace detail {

template <size_t OutputType, size_t InputType>
struct ConvertFunctor {
    static void function(cudaStream_t stream,
                         size_t size,
                         void* output,
                         const void* input,
                         unsigned numBlocks,
                         unsigned threadsPerBlock) {
        constexpr Type_t output_type = static_cast<Type_t>(OutputType + type_t_first_value);
        constexpr Type_t input_type = static_cast<Type_t>(InputType + type_t_first_value);
        using TOutput = cuda_type_traits_t<output_type>;
        using TInput = cuda_type_traits_t<input_type>;
        if (OutputType == InputType) {
            if (output == input) return;
            throwIfError(cudaMemcpyAsync(output, input, size * sizeof(TOutput), cudaMemcpyDeviceToDevice, stream));
        } else {
            CUDAPlugin::kernel::convert_impl<TOutput, TInput><<<numBlocks, threadsPerBlock, 0, stream>>>(
                size, static_cast<TOutput*>(output), static_cast<const TInput*>(input));
        }
    }
};

using convert_t = Convert::convert_t;

constexpr size_t type_count = type_t_last_value - type_t_first_value + 1;

template <template <size_t> class Template>
struct convert_vector : std::array<convert_t, type_count> {
    constexpr convert_vector() : convert_vector(std::make_index_sequence<type_count>()) {}

private:
    template <size_t... I>
    constexpr convert_vector(std::index_sequence<I...>)
        : std::array<convert_t, type_count>{&Template<I>::function...} {}
};

template <template <size_t, size_t> class Template, size_t N>
struct reduce {
    template <size_t M>
    using type = Template<N, M>;
};

template <template <size_t, size_t> class Template>
class convert_matrix : public std::array<std::array<convert_t, type_count>, type_count> {
public:
    constexpr convert_matrix() : convert_matrix<Template>(std::make_index_sequence<type_count>()) {}

private:
    template <size_t... I>
    constexpr convert_matrix(std::index_sequence<I...>)
        : std::array<std::array<convert_t, type_count>, type_count>{
              convert_vector<reduce<Template, I>::template type>{}...} {}
};
}  // namespace detail

Convert::Convert(
    Type_t output_element_type, Type_t input_element_type, size_t size, size_t numBlocks, size_t threadsPerBlock)
    : size_{size}, num_blocks_{numBlocks}, threads_per_block_{threadsPerBlock} {
    convert_kernel_ = getConvertKernel(output_element_type, input_element_type);
}

Convert::convert_t Convert::getConvertKernel(Type_t output_element_type, Type_t input_element_type) {
    static constexpr detail::convert_matrix<detail::ConvertFunctor> matrix{};
    const size_t input_type_index = static_cast<size_t>(input_element_type) - type_t_first_value;
    const size_t output_type_index = static_cast<size_t>(output_element_type) - type_t_first_value;
    return matrix[output_type_index][input_type_index];
}

void Convert::operator()(cudaStream_t stream, void* output, const void* src) const {
    convert_kernel_(stream, size_, output, src, num_blocks_, threads_per_block_);
}

}  // namespace kernel
}  // namespace CUDAPlugin
