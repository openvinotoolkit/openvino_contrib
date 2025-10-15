// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda/runtime.hpp>

void enqueueVecAdd(const CUDA::Stream s, dim3 gridDim, dim3 blockDim,
                   int* devA, int* devB, int* devC, int N);