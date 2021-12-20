// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "runtime.hpp"

namespace CUDA {

class Event : public Handle<cudaEvent_t> {
public:
    Event() : Handle(static_cast<__host__ cudaError_t (*)(cudaEvent_t* event)>(cudaEventCreate), cudaEventDestroy) {}
    auto&& record(const Stream& stream) {
        throwIfError(cudaEventRecord(get(), stream.get()));
        return std::move(*this);
    }
    float elapsedSince(const Event& start) const { return create(cudaEventElapsedTime, start.get(), get()); }
};

}  // namespace CUDA
