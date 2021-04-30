// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_memory_model.hpp"

namespace CUDAPlugin {

size_t MemoryModel::getDeviceMemoryBlockSize() {
    // TODO: Should be added proper implementation
    return 0;
}

ptrdiff_t MemoryModel::getOffsetForTensorId(unsigned id) {
    // TODO: Should be added proper implementation
    return 0;
}

} // namespace CUDAPlugin
