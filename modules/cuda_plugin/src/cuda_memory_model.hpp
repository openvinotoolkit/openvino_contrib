// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

namespace CUDAPlugin {

class MemoryModel {
 public:
    using Ptr = std::shared_ptr<MemoryModel>;
    size_t getDeviceMemoryBlockSize();
    ptrdiff_t getOffsetForTensorId(unsigned id);
};

} // namespace CUDAPlugin
