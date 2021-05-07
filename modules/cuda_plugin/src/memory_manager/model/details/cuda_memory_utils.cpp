// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_manager/model/cuda_memory_model.hpp"

namespace CUDAPlugin {

size_t applyAllignment(size_t value) {
  const size_t allignment = 256;
  return (value % allignment) == 0 ? value : value - (value % allignment) + allignment;
}

}  // namespace CUDAPlugin
