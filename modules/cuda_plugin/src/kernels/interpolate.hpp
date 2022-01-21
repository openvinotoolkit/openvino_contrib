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
    /// \brief Round modes for the nearest interpolation.
    enum class NearestMode { round_prefer_floor, round_prefer_ceil, floor, ceil, simple };

    enum class TransformMode { half_pixel, pytorch_half_pixel, asymmetric, tf_half_pixel_for_nn, align_corners };

    Interpolate(size_t num_blocks,
                size_t threads_per_block,
                CUDAPlugin::kernel::Type_t element_type,
                bool upscale,
                NearestMode nearest_mode,
                TransformMode transform_mode);

    void operator()(const cudaStream_t stream,
                    const void* src,
                    const size_t* input_strides,
                    const size_t* output_strides,
                    const float* slices,
                    const size_t* input_shape,
                    const size_t* output_shape,
                    void* dst) const;

private:
    template <typename T>
    void callKernel(const cudaStream_t stream,
                    const void* src,
                    const size_t* input_strides,
                    const size_t* output_strides,
                    const float* slices,
                    const size_t* input_shape,
                    const size_t* output_shape,
                    void* dst) const;

private:
    size_t num_blocks_;
    size_t threads_per_block_;
    Type_t element_type_;
    bool use_optimized_kernel_;
    NearestMode nearest_mode_;
    TransformMode transform_mode_;
};

}  // namespace kernel
}  // namespace CUDAPlugin
