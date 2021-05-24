// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "runtime.hpp"
#include <cutensor.h>

namespace CUDA {
namespace cutensor::details {
constexpr inline cudaError_t dummyDestroy(cutensorHandle_t) { return cudaSuccess; }
}

class CuTensorHandle : public UniqueBase<cutensorInit, cutensor::details::dummyDestroy> {
};

}  // namespace CUDA
