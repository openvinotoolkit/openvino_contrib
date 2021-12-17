// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_fp16.h>

#include "convert.cuh"
#include "kernels/range.hpp"
#include "typed_functor.hpp"

namespace CUDAPlugin {

namespace kernel {

template <typename T_IN1, typename T_IN2, typename T_OUT>
static __global__ typename std::enable_if<std::is_same<T_OUT, __half>::value>::type range(const T_IN1* start,
                                                                                          const T_IN2* step,
                                                                                          const size_t dstSize,
                                                                                          T_OUT* dst) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dstSize) {
        return;
    }
    dst[i] = __hadd(cast<T_OUT, T_IN1>(start[0]),
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
                    __hmul(cast<T_OUT, T_IN2>(step[0]), cast<T_OUT, decltype(i)>(i))
#else
                    // the __hmul operation isn't supported here. Also, operation+ and operation* aren't defined here
                    cast<T_OUT, float>(cast<float, T_IN2>(step[0]) * cast<float, decltype(i)>(i))
#endif
    );
}

template <typename T_IN1, typename T_IN2, typename T_OUT>
static __global__ typename std::enable_if<!std::is_same<T_OUT, __half>::value>::type range(const T_IN1* start,
                                                                                           const T_IN2* step,
                                                                                           const size_t dstSize,
                                                                                           T_OUT* dst) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dstSize) {
        return;
    }
    dst[i] = cast<T_OUT>(start[0]) + cast<T_OUT>(step[0]) * cast<T_OUT>(i);
}

template <typename T_IN1, typename T_IN2, typename T_OUT>
struct RangeFunctor {
    static void function(cudaStream_t stream,
                         unsigned blocks_number,
                         unsigned threads_per_block,
                         const void* start,
                         const void* step,
                         const size_t dstSize,
                         void* dst) {
        range<T_IN1, T_IN2, T_OUT><<<blocks_number, threads_per_block, 0, stream>>>(
            static_cast<const T_IN1*>(start), static_cast<const T_IN2*>(step), dstSize, static_cast<T_OUT*>(dst));
    }
};

RangeKernelOp::RangeKernelOp(const size_t max_size,
                             const unsigned blocks_number,
                             const unsigned threads_per_block,
                             const Type_t input_start_type,
                             const Type_t input_stop_type,
                             const Type_t input_step_type,
                             const Type_t output_type)
    : blocks_number_{blocks_number}, threads_per_block_{threads_per_block} {
    static constexpr TypedFunctor<RangeFunctor, TFuncPtr, DIM_3D> combinations{};
    func_ptr_ = combinations[input_start_type][input_step_type][output_type];
}

void RangeKernelOp::operator()(
    const cudaStream_t stream, const void* start, const void* step, const size_t dstSize, void* dst) const {
    func_ptr_(stream, blocks_number_, threads_per_block_, start, step, dstSize, dst);
}

}  // namespace kernel

}  // namespace CUDAPlugin
