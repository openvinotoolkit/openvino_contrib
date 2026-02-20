// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert.cuh"
#include "convert.hpp"
#include "details/error.hpp"
#include "details/type_validator.hpp"
#include "details/typed_functor.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

template <typename TOutput, typename TInput>
__global__ void convert_impl(size_t inputSize, TOutput* out, const TInput* in) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < inputSize) {
        out[i] = cast<TOutput>(in[i]);
    }
}

template <typename TOutput, typename TInput, typename sfinae_helper = void>
struct ConvertFunctor;

template <typename TOutput, typename TInput>
struct ConvertFunctor<TOutput, TInput, typename std::enable_if<std::is_same<TOutput, TInput>::value>::type> {
    static void function(cudaStream_t stream,
                         size_t size,
                         void* output,
                         const void* input,
                         unsigned /*numBlocks*/,
                         unsigned /*threadsPerBlock*/) {
        if (output == input) return;
        throwIfError(cudaMemcpyAsync(output, input, size * sizeof(TOutput), cudaMemcpyDeviceToDevice, stream));
    }
};

template <typename TOutput, typename TInput>
struct ConvertFunctor<TOutput, TInput, typename std::enable_if<!std::is_same<TOutput, TInput>::value>::type> {
    static void function(cudaStream_t stream,
                         size_t size,
                         void* output,
                         const void* input,
                         unsigned numBlocks,
                         unsigned threadsPerBlock) {
        ov::nvidia_gpu::kernel::convert_impl<TOutput, TInput><<<numBlocks, threadsPerBlock, 0, stream>>>(
            size, static_cast<TOutput*>(output), static_cast<const TInput*>(input));
    }
};

Convert::Convert(
    Type_t output_element_type, Type_t input_element_type, size_t size, size_t numBlocks, size_t threadsPerBlock)
    : size_{size}, num_blocks_{numBlocks}, threads_per_block_{threadsPerBlock} {
    TypeValidator<AllElementTypesSwitch>::check(output_element_type);
    TypeValidator<AllElementTypesSwitch>::check(input_element_type);
    static constexpr TypedFunctor<ConvertFunctor, convert_t, DIM_2D> combinations{};
    convert_kernel_ = combinations[output_element_type][input_element_type];
}

void Convert::operator()(cudaStream_t stream, void* output, const void* src) const {
    convert_kernel_(stream, size_, output, src, num_blocks_, threads_per_block_);
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
