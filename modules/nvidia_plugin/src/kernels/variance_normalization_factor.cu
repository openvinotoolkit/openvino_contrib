// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda/math.cuh>

#include "convert.cuh"
#include "kernels/variance_normalization_factor.hpp"
#include "typed_functor.hpp"

namespace ov {
namespace nvidia_gpu {

namespace kernel {

template <typename T, bool epsilon_inside_sqrt>
static __global__ void varianceNormalizationFactor(double epsilon, size_t data_size, T *data) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= data_size) {
        return;
    }
    if (epsilon_inside_sqrt) {
        data[i] = CUDA::math::pow(CUDA::math::sqrt(data[i] + cast<T>(epsilon)), cast<T>(-1));
    } else {
        data[i] = CUDA::math::pow(CUDA::math::sqrt(data[i]) + cast<T>(epsilon), cast<T>(-1));
    }
}

template <typename T, bool epsilon_inside_sqrt>
void kernelCaller(cudaStream_t stream,
                  unsigned blocks_number,
                  unsigned threads_per_block,
                  double epsilon,
                  size_t data_size,
                  void *data) {
    varianceNormalizationFactor<T, epsilon_inside_sqrt>
        <<<blocks_number, threads_per_block, 0, stream>>>(epsilon, data_size, static_cast<T *>(data));
}

VarianceNormalizationFactor::VarianceNormalizationFactor(unsigned blocks_number,
                                                         unsigned threads_per_block,
                                                         double epsilon,
                                                         size_t data_size,
                                                         Type_t data_type,
                                                         bool epsilon_inside_sqrt)
    : blocks_number_{blocks_number}, threads_per_block_{threads_per_block}, epsilon_{epsilon}, data_size_{data_size} {
#define CASE(type)                                                                               \
    case Type_t::type: {                                                                         \
        func_ptr_ = epsilon_inside_sqrt ? kernelCaller<cuda_type_traits_t<Type_t::type>, true>   \
                                        : kernelCaller<cuda_type_traits_t<Type_t::type>, false>; \
        break;                                                                                   \
    }

    switch (data_type) {
#ifdef CUDA_HAS_BF16_TYPE
        CASE(bf16)
#endif
        CASE(f16)
        CASE(f32)
        CASE(f64)
        default:
            throwIEException(fmt::format("ov::nvidia_gpu::MvnOp: unsupported data type, must be any float point type."));
    }
#undef CASE
}

void VarianceNormalizationFactor::operator()(cudaStream_t stream, void *data) const {
    func_ptr_(stream, blocks_number_, threads_per_block_, epsilon_, data_size_, data);
}

}  // namespace kernel

}  // namespace nvidia_gpu
}  // namespace ov
