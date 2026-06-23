// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <torch/extension.h>

// Check if CUDA is available
#ifdef __CUDACC__
#define CUDA_AVAILABLE 1
#else
#define CUDA_AVAILABLE 0
#endif

#if CUDA_AVAILABLE
#include <ATen/cuda/CUDAContext.h>
#endif

#define CHECK_CONTIGUOUS(x)                                         \
  do {                                                              \
    TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor"); \
  } while (0)

#define CHECK_IS_INT(x)                              \
  do {                                               \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, \
             #x " must be an int tensor");           \
  } while (0)

#define CHECK_IS_FLOAT(x)                              \
  do {                                                 \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, \
             #x " must be a float tensor");            \
  } while (0)

#if CUDA_AVAILABLE
#define CHECK_CUDA(x)                                          \
  do {                                                         \
    TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor"); \
  } while (0)
#else
// CPU-only version - provide empty macro for CHECK_CUDA
#define CHECK_CUDA(x) do {} while (0)
#endif
