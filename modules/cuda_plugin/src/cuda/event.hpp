// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_runtime_api.h>
#include "runtime.hpp"
#include "stream.hpp"

namespace CUDA {

class Event {
  /* : public UniqueBase<&cudaEventCreate, &cudaEventDestroy, cudaEvent_t>
   *   does not work: unable to deduce ‘auto’ from ‘& cudaEventCreate’  */
public:
  using Native = cudaEvent_t;
  Event(Event&& that) : native { that.native } { that.native = {}; }
  Event operator=(const Event&) = delete;
  explicit Event(const Stream& stream) {
    logIfError(cudaEventCreate(&native));
    logIfError(cudaEventRecord(get(), stream.get()));
  }
  ~Event() {
    if (native != Native{}) logIfError(cudaEventDestroy(native));
  }
  Native get() const noexcept { return native; }
  float elapsedSince(const Event& start) const {
    float result {std::numeric_limits<float>::quiet_NaN()};
    logIfError(cudaEventElapsedTime(&result, start.get(), get()));
    return result;
  }
private:
  Native native;
};
} //namespace CUDA
