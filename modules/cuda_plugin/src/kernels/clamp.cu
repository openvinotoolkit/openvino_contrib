// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "clamp.hpp"
#include "elementwise.cuh"
#include "error.hpp"
#include "tensor_helpers.hpp"

namespace CUDAPlugin {
namespace kernel {

template <typename T>
struct ClampOpImpl {
    __device__ static inline T op(T x, T min_value, T max_value) {
        return kernel::min(kernel::max(x, min_value), max_value);
    }
};

template <typename ElementTypes>
struct MinMaxSwitch {
    template <typename T>
    void case_(Type_t element_type,
               size_t max_threads_per_block,
               cudaStream_t stream,
               const void* in,
               size_t num_elements,
               void* out,
               double d_min,
               double d_max) const {
        T min = double_round_cast<T>(d_min, std::ceil);
        T max = double_round_cast<T>(d_max, std::floor);

        using EW = ElementwiseUnary<AllElementTypesSwitch, ClampOpImpl>;
        EW ew{element_type, max_threads_per_block};
        ew.callKernel<T>(stream, in, num_elements, out, min, max);
    }

    template <typename T>
    void default_(T t,
                  Type_t element_type,
                  size_t max_threads_per_block,
                  cudaStream_t stream,
                  const void* in,
                  size_t num_elements,
                  void* out,
                  double d_min,
                  double d_max) const {
        throwIEException(fmt::format("Element type = {} is not supported.", t));
    }
};

Clamp::Clamp(Type_t element_type, size_t max_threads_per_block)
    : element_type_{element_type}, max_threads_per_block_{max_threads_per_block} {}

void Clamp::operator()(
    cudaStream_t stream, const void* in, size_t num_elements, void* out, double min, double max) const {
    MinMaxSwitch<AllElementTypesSwitch> switcher{};
    AllElementTypesSwitch::switch_(
        element_type_, switcher, element_type_, max_threads_per_block_, stream, in, num_elements, out, min, max);
}

}  // namespace kernel
}  // namespace CUDAPlugin
