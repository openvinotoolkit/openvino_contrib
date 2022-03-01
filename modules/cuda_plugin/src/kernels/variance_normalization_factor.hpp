// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_runtime.h>

#include "cuda_type_traits.hpp"
#include "ngraph/type/element_type.hpp"

namespace CUDAPlugin {
namespace kernel {

class VarianceNormalizationFactor {
public:
    VarianceNormalizationFactor(unsigned blocks_number,
                                unsigned threads_per_block,
                                double epsilon,
                                size_t data_size,
                                Type_t data_type,
                                bool epsilon_inside_sqrt);

    void operator()(cudaStream_t stream, void *data) const;

private:
    unsigned blocks_number_;
    unsigned threads_per_block_;
    double epsilon_;
    size_t data_size_;
    using TFuncPtr = void (*)(cudaStream_t, unsigned, unsigned, double, size_t, void *);
    TFuncPtr func_ptr_;
};

}  // namespace kernel

}  // namespace CUDAPlugin
