// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_runtime.h>

namespace ov {
namespace nvidia_gpu {
namespace kernel {

/// Simple product-reduction kernel for INT32 (cuDNN reduceTensor does not support
/// integer types). This performs a FULL reduction only — all `num_elements` inputs
/// are multiplied into a single scalar `output[0]`. It takes no axes argument and
/// does not support partial-axis reductions, so the caller must guarantee a
/// full-axis reduction to a single element (ReduceProdIntOp asserts this) and that
/// num_elements fits in a single block.
void reduce_prod_int32(cudaStream_t stream,
                       const int32_t* input,
                       int32_t* output,
                       size_t num_elements);

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
