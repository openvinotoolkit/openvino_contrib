// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda/math.cuh>

#include "clamp.hpp"
#include "error.hpp"
#include "tensor_helpers.hpp"

namespace CUDAPlugin {
namespace kernel {

namespace cumath = CUDA::math;

template <typename T>
struct ClampOpImpl {
    __device__ static inline T op(T x, T min_value, T max_value) {
        return cumath::min(cumath::max(x, min_value), max_value);
    }
};

template <typename ElementTypes>
struct MinMaxSwitch {
    template <typename T>
    void case_(
        const EWClamp& ewclamp, cudaStream_t stream, const void* in, void* out, double d_min, double d_max) const {
        T min = double_round_cast<T>(d_min, std::ceil);
        T max = double_round_cast<T>(d_max, std::floor);
        ewclamp.callKernel<T>(stream, in, out, min, max);
    }

    template <typename T>
    void default_(T t, const EWClamp&, cudaStream_t, const void*, void*, double, double) const {
        throwIEException(fmt::format("Element type = {} is not supported.", t));
    }
};

Clamp::Clamp(Type_t element_type, size_t max_threads_per_block, size_t num_elements, double min, double max)
    : element_type_{element_type},
      ew_clamp_{element_type_, max_threads_per_block, num_elements},
      min_{min},
      max_{max} {}

void Clamp::operator()(cudaStream_t stream, const void* in, void* out) const {
    MinMaxSwitch<AllElementTypesSwitch> switcher{};
    AllElementTypesSwitch::switch_(element_type_, switcher, ew_clamp_, stream, in, out, min_, max_);
}

}  // namespace kernel
}  // namespace CUDAPlugin
