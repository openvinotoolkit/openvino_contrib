// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "cuda_type_traits.hpp"
#include "error.hpp"

namespace CUDAPlugin {
namespace kernel {

class Interpolate {
public:
    Interpolate(size_t num_blocks, size_t threads_per_block, CUDAPlugin::kernel::Type_t element_type, bool upscale);
    void operator()(const cudaStream_t stream,
                    const void* src,
                    const size_t* input_strides,
                    const size_t* output_strides,
                    const float* slices,
                    void* dst) const;

private:
    template <typename T>
    void callKernel(const cudaStream_t stream,
                    const void* src,
                    const size_t* input_strides,
                    const size_t* output_strides,
                    const float* slices,
                    void* dst) const;

private:
    size_t num_blocks_;
    size_t threads_per_block_;
    Type_t element_type_;
    bool upscale_;
};

}  // namespace kernel
}  // namespace CUDAPlugin
