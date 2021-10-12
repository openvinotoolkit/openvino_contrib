// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "runtime.hpp"

namespace CUDA {
class Event : public UniqueBase<static_cast<__host__ cudaError_t (*)(cudaEvent_t* event)>(cudaEventCreate),
                                cudaEventDestroy,
                                cudaEvent_t> {
public:
    explicit Event(const Stream& stream) { throwIfError(cudaEventRecord(get(), stream.get())); }
    float elapsedSince(const Event& start) const { return create(cudaEventElapsedTime, start.get(), get()); }
};

}  // namespace CUDA
