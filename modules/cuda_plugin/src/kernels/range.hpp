// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_runtime.h>

#include "cuda_type_traits.hpp"
#include "ngraph/type/element_type.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

class RangeKernelOp {
public:
    RangeKernelOp(size_t max_size,
                  unsigned blocks_number,
                  unsigned threads_per_block,
                  Type_t input_start_type,
                  Type_t input_stop_type,
                  Type_t inputStep_type,
                  Type_t output_type);

    void operator()(cudaStream_t stream, const void *start, const void *step, size_t dstSize, void *dst) const;

private:
    using TFuncPtr = void (*)(cudaStream_t, unsigned, unsigned, const void *, const void *, const size_t, void *);
    TFuncPtr func_ptr_;
    unsigned blocks_number_;
    unsigned threads_per_block_;
};

}  // namespace kernel

}  // namespace nvidia_gpu
}  // namespace ov
