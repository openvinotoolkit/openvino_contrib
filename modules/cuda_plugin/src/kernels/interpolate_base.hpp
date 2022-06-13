// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "tensor_helpers.hpp"

namespace CUDAPlugin {
namespace kernel {

class InterpolateBase {
public:
    enum class CoordinateTransformMode {
        half_pixel,
        pytorch_half_pixel,
        asymmetric,
        tf_half_pixel_for_nn,
        align_corners
    };

    static constexpr unsigned MAX_SHAPE_RANK = 5;
    using UIntShape = Shape<unsigned, MAX_SHAPE_RANK>;
    using FloatShape = Shape<float, MAX_SHAPE_RANK>;
    using IntShape = Shape<int, MAX_SHAPE_RANK>;
    struct Index {
        IntShape v{};
    };

    class details;
};

}  // namespace kernel
}  // namespace CUDAPlugin
