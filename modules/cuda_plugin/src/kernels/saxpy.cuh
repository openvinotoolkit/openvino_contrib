// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace CUDAPlugin {

extern __global__ void saxpy(int n, const float *a, const float *x, float *y);

} // namespace CUDAPlugin
