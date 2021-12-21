// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda_type_traits.hpp"

namespace CUDAPlugin {
namespace kernel {

class Clamp {
public:
    Clamp(Type_t element_type, size_t max_threads_per_block);
    Clamp(Clamp&&) = default;
    Clamp& operator=(Clamp&&) = default;

    void operator()(cudaStream_t stream, const void* in, size_t num_elements, void* out, double min, double max) const;

private:
    Type_t element_type_;
    size_t max_threads_per_block_;
};

}  // namespace kernel
}  // namespace CUDAPlugin
