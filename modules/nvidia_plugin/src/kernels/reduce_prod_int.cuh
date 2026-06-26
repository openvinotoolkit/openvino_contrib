// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_runtime.h>

namespace ov {
namespace nvidia_gpu {
namespace kernel {

/// Simple reduction kernel for integer types (INT32/UINT32).
/// cuDNN reduceTensor does not support integer types, so we use a custom kernel.
/// The kernel reduces input tensor to output tensor along specified axes.
/// For simplicity, this implements a full reduction (all elements → single scalar).
void reduce_prod_int32(cudaStream_t stream,
                       const int32_t* input,
                       int32_t* output,
                       size_t num_elements);

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
