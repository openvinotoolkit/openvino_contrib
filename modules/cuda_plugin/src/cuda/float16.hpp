// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>

#if CUDA_VERSION >= 11000
#define CUDA_HAS_BF16_TYPE
#include <cuda_bf16.h>
#endif  // CUDA_VERSION >= 11000

#ifdef __CUDACC__

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
#define CUDA_HAS_HALF_MATH
#endif  // (__CUDA_ARCH__ >= 530) || !defined(__CUDA_ARCH__)

#if defined(CUDA_HAS_BF16_TYPE) && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
#define CUDA_HAS_BF16_MATH
#endif  // defined (CUDA_HAS_BF16_TYPE) && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))

#endif  // __CUDACC__
