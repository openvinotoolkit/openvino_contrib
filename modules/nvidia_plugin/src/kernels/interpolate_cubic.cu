// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <cuda/math.cuh>
#include <gsl/gsl_assert>

#include "interpolate_cubic.hpp"
#include "interpolate_details.cuh"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

template <typename CT>
static inline __device__ void get_cubic_coeff(CT s, CT a, CT coeff[4]) {
    CT abs_s = CUDA::math::abs(s);
    const CT one{1.0f};
    const CT two{2.0f};
    const CT three{3.0f};
    const CT four{4.0f};
    const CT five{5.0f};
    const CT eight{8.0f};
    coeff[0] = ((a * (abs_s + one) - five * a) * (abs_s + one) + eight * a) * (abs_s + one) - four * a;
    coeff[1] = ((a + two) * abs_s - (a + three)) * abs_s * abs_s + one;
    coeff[2] = ((a + two) * (one - abs_s) - (a + three)) * (one - abs_s) * (one - abs_s) + one;
    coeff[3] = ((a * (two - abs_s) - five * a) * (two - abs_s) + eight * a) * (two - abs_s) - four * a;
}

static inline __device__ unsigned clip_coord(const InterpolateCubic::Props* props, int coord, int axis) {
    return max(0, min(coord, static_cast<int>(props->input_shape[axis]) - 1));
}

template <typename T, typename ComputeType>
static __global__ void interpolate_cubic(const T* input,
                                         T* output,
                                         const InterpolateCubic::Props* props,
                                         const InterpolateCubic::Index* indices,
                                         const unsigned num_of_indices) {
    const unsigned output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= shape_size(props->output_shape)) return;

    using CT = ComputeType;
    using details = InterpolateCubic::details;
    using IntShape = InterpolateCubic::IntShape;
    using UIntShape = InterpolateCubic::UIntShape;

    UIntShape output_coords{};
    shape_indices(props->output_shape, output_idx, output_coords);

    IntShape input_coords;
    details::shape_copy(output_coords, input_coords);
    const CT cube_coeff{props->cube_coeff};
    CT cubic_coeffs[InterpolateCubic::MAX_SHAPE_RANK][4]{};
    for (unsigned i = 0; i < props->num_of_axes; ++i) {
        const unsigned axis = props->axes[i];
        const CT in_coord = details::get_original_coordinate<CT>(props->transform_mode,
                                                                 output_coords[axis],
                                                                 static_cast<CT>(props->scales[i]),
                                                                 props->output_shape[axis],
                                                                 props->input_shape[axis]);
        const CT in_coord_int = CUDA::math::floor(in_coord);
        input_coords[axis] = static_cast<int>(in_coord_int);
        get_cubic_coeff(in_coord - in_coord_int, cube_coeff, cubic_coeffs[axis]);
    }

    UIntShape coords_for_sum;
    details::shape_copy(input_coords, coords_for_sum);

    CT summa{};
    for (unsigned index_i = 0; index_i < num_of_indices; ++index_i) {
        const IntShape& index = indices[index_i].v;
        CT coeffs_prod{1.0f};
        for (unsigned i = 0; i < props->num_of_axes; ++i) {
            const unsigned axis = props->axes[i];
            coords_for_sum[axis] = clip_coord(props, input_coords[axis] + index[i] - 1, axis);
            coeffs_prod = coeffs_prod * cubic_coeffs[axis][index[i]];
        }
        const unsigned input_index = flat_address_by_shape(props->input_shape, coords_for_sum);
        summa += coeffs_prod * static_cast<CT>(input[input_index]);
    }

    output[output_idx] = static_cast<T>(summa);
}

InterpolateCubic::InterpolateCubic(std::vector<size_t> in_shape,
                                   std::vector<size_t> axes,
                                   std::vector<float> scales,
                                   std::vector<size_t> out_shape,
                                   CoordinateTransformMode transform_mode,
                                   float cube_coeff,
                                   Type_t element_type,
                                   size_t max_threads_per_block)
    : props_device_ptr_{nullptr},
      indices_device_ptr_{nullptr},
      num_blocks_{},
      threads_per_block_{},
      element_type_{element_type} {
    Expects(in_shape.size() == out_shape.size());
    Expects(in_shape.size() <= MAX_SHAPE_RANK);
    Expects(axes.size() == scales.size());

    std::copy(in_shape.begin(), in_shape.end(), props_.input_shape);
    std::copy(axes.begin(), axes.end(), props_.axes);
    std::copy(scales.begin(), scales.end(), props_.scales);
    std::copy(out_shape.begin(), out_shape.end(), props_.output_shape);

    props_.num_of_axes = axes.size();
    props_.cube_coeff = cube_coeff;
    props_.transform_mode = transform_mode;

    std::vector<int> indices_shape(props_.num_of_axes, 4);
    details::ShapeIterator iter{indices_shape};
    while (!iter.end()) {
        Index index;
        std::copy(iter.value().begin(), iter.value().end(), &index.v[0]);
        indices_.push_back(index);
        iter.advance();
    }

    std::tie(num_blocks_, threads_per_block_) =
        calculateElementwiseGrid(shape_size(props_.output_shape), max_threads_per_block);
}

void InterpolateCubic::operator()(const cudaStream_t stream, const void* input, void* output) const {
    switch (element_type_) {
        case Type_t::f16:
            return callKernel<__half, __half>(stream, input, output);
        case Type_t::f32:
            return callKernel<float, float>(stream, input, output);
#ifdef CUDA_HAS_BF16_TYPE
        case Type_t::bf16:
            return callKernel<__nv_bfloat16, __nv_bfloat16>(stream, input, output);
#endif
        case Type_t::i8:
            return callKernel<int8_t, float>(stream, input, output);
        case Type_t::u8:
            return callKernel<uint8_t, float>(stream, input, output);
        default:
            throwIEException(
                fmt::format("Element type = {} is not supported by InterpolateCubic operation.", element_type_));
    }
}

template <typename T, typename CT>
void InterpolateCubic::callKernel(const cudaStream_t stream, const void* input, void* output) const {
    kernel::interpolate_cubic<T, CT>
        <<<num_blocks_, threads_per_block_, 0, stream>>>(static_cast<const T*>(input),
                                                         static_cast<T*>(output),
                                                         static_cast<const Props*>(props_device_ptr_),
                                                         static_cast<const Index*>(indices_device_ptr_),
                                                         indices_.size());
}

std::vector<size_t> InterpolateCubic::immutableWorkbufferSizes() const {
    return {sizeof(Props), sizeof(Index) * indices_.size()};
}

void InterpolateCubic::initImmutableWorkbuffers(const std::vector<void*>& buffers) {
    kernel::throwIfError(
        cudaMemcpyAsync(buffers[0], static_cast<const void*>(&props_), sizeof(props_), cudaMemcpyHostToDevice));
    props_device_ptr_ = buffers[0];

    kernel::throwIfError(cudaMemcpyAsync(
        buffers[1], static_cast<const void*>(&indices_[0]), sizeof(Index) * indices_.size(), cudaMemcpyHostToDevice));
    indices_device_ptr_ = buffers[1];
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
