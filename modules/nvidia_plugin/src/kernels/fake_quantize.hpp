// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_runtime.h>

#include "cuda_type_traits.hpp"
#include "numpy_broadcast_mapper.cuh"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

class FakeQuantize {
public:
    FakeQuantize(Type_t element_type, std::size_t max_size, std::size_t threads_per_block, std::size_t levels);
    FakeQuantize(FakeQuantize&&) = default;
    FakeQuantize& operator=(FakeQuantize&&) = default;

    void operator()(const cudaStream_t stream,
                    const void* arg,
                    const void* in_low,
                    const void* in_high,
                    const void* out_low,
                    const void* out_high,
                    const NumpyBroadcastMapper& in_low_mapper,
                    const NumpyBroadcastMapper& in_high_mapper,
                    const NumpyBroadcastMapper& out_low_mapper,
                    const NumpyBroadcastMapper& out_high_mapper,
                    void* out) const;

private:
    template <typename T>
    void Call(const cudaStream_t stream,
              const void* arg,
              const void* in_low,
              const void* in_high,
              const void* out_low,
              const void* out_high,
              const NumpyBroadcastMapper& in_low_mapper,
              const NumpyBroadcastMapper& in_high_mapper,
              const NumpyBroadcastMapper& out_low_mapper,
              const NumpyBroadcastMapper& out_high_mapper,
              T levels_1,
              void* out) const;

    Type_t element_type_{};
    std::size_t max_size_{};
    std::size_t num_blocks_{};
    std::size_t threads_per_block_{};
    std::size_t levels_{};
};

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
