// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <details/ie_exception.hpp>
#include "stream.hpp"

namespace CUDAPlugin {

CudaStream::CudaStream() {
    auto err = cudaStreamCreate(&stream_);
    if (cudaSuccess != err) {
        THROW_IE_EXCEPTION << "Internal error: " << cudaGetErrorString(err);
    }
}

CudaStream::CudaStream(unsigned int flags) {
    auto err = cudaStreamCreateWithFlags(&stream_, flags);
    if (cudaSuccess != err) {
        THROW_IE_EXCEPTION << "Internal error: " << cudaGetErrorString(err);
    }
}

CudaStream::CudaStream(unsigned int flags, int priority) {
    auto err = cudaStreamCreateWithPriority(&stream_, flags, priority);
    if (cudaSuccess != err) {
        THROW_IE_EXCEPTION << "Internal error: " << cudaGetErrorString(err);
    }
}

CudaStream::~CudaStream() {
    cudaStreamDestroy(stream_);
}

void CudaStream::synchronize() {
    auto err = cudaStreamSynchronize(stream_);
    if (cudaSuccess != err) {
        THROW_IE_EXCEPTION << "Internal error: " << cudaGetErrorString(err);
    }
}

} // namespace CUDAPlugin
