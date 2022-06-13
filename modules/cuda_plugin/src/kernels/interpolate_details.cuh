// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <assert.h>

#include <vector>

#include "convert.cuh"
#include "interpolate_base.hpp"

namespace CUDAPlugin {
namespace kernel {

class InterpolateBase::details {
public:
    class ShapeIterator {
        std::vector<int> shape_;
        std::vector<int> value_;

    public:
        ShapeIterator(const std::vector<int>& shape) : shape_{shape}, value_(shape.size(), 0) {}

        const std::vector<int>& value() const { return value_; }

        void advance() {
            for (int i = value_.size() - 1; i >= 0; --i) {
                ++value_[i];
                if (value_[i] < shape_[i]) break;

                if (i == 0) {
                    value_.clear();
                    break;
                }

                value_[i] = 0;
            }
        }

        bool end() const { return value_.empty(); }
    };

    template <typename CT>
    static inline __device__ CT get_original_coordinate(const CoordinateTransformMode mode,
                                                        const unsigned output_coordinate,
                                                        const CT scale,
                                                        const unsigned output_dim,
                                                        const unsigned input_dim) {
        CT input_coord{};
        switch (mode) {
            case CoordinateTransformMode::half_pixel:
                input_coord = ((cast<CT>(output_coordinate) + CT{0.5f}) / scale) - CT{0.5f};
                break;
            case CoordinateTransformMode::pytorch_half_pixel:
                if (output_dim > 1)
                    input_coord = ((cast<CT>(output_coordinate) + CT{0.5f}) / scale) - CT{0.5f};
                else
                    input_coord = CT{};
                break;
            case CoordinateTransformMode::asymmetric:
                input_coord = cast<CT>(output_coordinate) / scale;
                break;
            case CoordinateTransformMode::tf_half_pixel_for_nn:
                input_coord = (cast<CT>(output_coordinate) + CT{0.5f}) / scale;
                break;
            case CoordinateTransformMode::align_corners:
                if (output_dim == 1)
                    input_coord = CT{};
                else
                    input_coord = cast<CT>(output_coordinate) * cast<CT>(input_dim - 1) / cast<CT>(output_dim - 1);
                break;
            default:
                assert(false);
        }
        return input_coord;
    }

    template <typename SRCT, typename DSTT, unsigned N>
    static inline __device__ void shape_copy(const Shape<SRCT, N>& src, Shape<DSTT, N>& dst) {
        for (int i = 0; i < N; ++i) dst[i] = cast<DSTT>(src[i]);
    }
};

}  // namespace kernel
}  // namespace CUDAPlugin
