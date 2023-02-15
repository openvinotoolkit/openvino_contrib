// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "details/cuda_type_traits.hpp"
#include "details/error.hpp"
#include "details/tensor_helpers.hpp"
#include "interpolate_base.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

class InterpolateLinear : public InterpolateBase {
public:
    InterpolateLinear(std::vector<size_t> in_shape,
                      std::vector<size_t> axes,
                      std::vector<float> scales,
                      std::vector<size_t> out_shape,
                      CoordinateTransformMode transform_mode,
                      bool antialias,
                      Type_t element_type,
                      size_t max_threads_per_block);

    void operator()(const cudaStream_t stream, const void* input, void* output) const;

    std::vector<size_t> immutableWorkbufferSizes() const;
    void initImmutableWorkbuffers(const std::vector<void*>& buffers);

    struct Props {
        UIntShape input_shape{};
        UIntShape output_shape{};

        unsigned num_of_axes{};
        UIntShape axes{};
        FloatShape scales{};
        FloatShape a{};
        IntShape r{};

        float prod_of_a{};
        CoordinateTransformMode transform_mode{};
    };

private:
    template <typename T, typename CT>
    void callKernel(const cudaStream_t stream, const void* input, void* output) const;

private:
    Props props_;
    const void* props_device_ptr_;
    std::vector<Index> indices_;
    const void* indices_device_ptr_;

    size_t num_blocks_;
    size_t threads_per_block_;
    Type_t element_type_;
};

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
