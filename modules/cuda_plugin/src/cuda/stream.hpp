// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_runtime_api.h>
#include <gpu/device_pointers.hpp>
#include <details/ie_no_copy.hpp>

namespace CUDAPlugin {

class CudaStream : protected InferenceEngine::details::no_copy {
 public:
    CudaStream();
    explicit CudaStream(unsigned int flags);
    CudaStream(unsigned int flags, int priority);
    ~CudaStream() override;

    /**
     * Copy memory asynchronous from Host to Device
     */
    template <typename TDes, typename TSrc>
    void memcpyAsync(InferenceEngine::gpu::DevicePointer<TDes*> dest, const TSrc* src, std::size_t n);
    /**
     * Copy memory asynchronous from Device to Host
     */
    template <typename TDes, typename TSrc>
    void memcpyAsync(TDes* dest, InferenceEngine::gpu::DevicePointer<const TSrc*> src, std::size_t n);

    /**
     * Blocks until \p stream has completed all operations.
     */
    void synchronize();

 private:
    cudaStream_t stream_;
};

template <typename TDes, typename TSrc>
void CudaStream::memcpyAsync(InferenceEngine::gpu::DevicePointer<TDes*> dest, const TSrc* src, std::size_t n) {
    auto err = cudaMemcpyAsync(dest, src, n, cudaMemcpyHostToDevice, stream_);
    if (cudaSuccess != err) {
        THROW_IE_EXCEPTION << "Internal error: " << cudaGetErrorString(err);
    }
}

template <typename TDes, typename TSrc>
void CudaStream::memcpyAsync(TDes* dest, InferenceEngine::gpu::DevicePointer<const TSrc*> src, std::size_t n) {
    auto err = cudaMemcpyAsync(dest, src, n, cudaMemcpyDeviceToHost, stream_);
    if (cudaSuccess != err) {
        THROW_IE_EXCEPTION << "Internal error: " << cudaGetErrorString(err);
    }
}

} // namespace CUDAPlugin
